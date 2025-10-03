# /kaggle/working/ARAA-Net/train_lasa_vgg.py
# ... (all imports and previous code) ...

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set seeds for reproducibility
    torch.manual_seed(42) # Use a fixed seed for general reproducibility
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(42)

    # --- Experiment Setup ---
    exp_name = f"{args.backbone}_LASA_Unet_FocalDice_DS_WaveletHE_{args.dataset_name.replace('TSRS_RSNA-', '').lower()}"
    exp_path = os.path.join(CKPT_ROOT, exp_name)
    check_mkdir(exp_path) # Create experiment directory if it doesn't exist
    setup_logging(exp_path) # Configure logging

    logging.info(f"Starting training for experiment: '{exp_name}'")
    logging.info(f"Using device: {device}")
    logging.info(f"Arguments: {args}")

    # --- Model Initialization ---
    net = LASA_Unet(num_classes=args.num_classes, backbone_name=args.backbone).to(device)
    logging.info(f"Model initialized: LASA-Unet with {args.backbone} backbone.")

    # --- Loss Functions ---
    focal_loss_fn = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma, ignore_index=255).to(device)
    dice_loss_fn = DiceLoss(smooth=1e-6, ignore_index=255).to(device)
    logging.info(f"Loss functions initialized: FocalLoss (alpha={args.focal_alpha}, gamma={args.focal_gamma}), DiceLoss (smooth={1e-6}).")

    # --- Optimizer and Scheduler ---
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=args.scheduler_factor, 
                                                     patience=args.scheduler_patience, min_lr=args.scheduler_min_lr, verbose=True)
    logging.info(f"Optimizer: Adam (lr={args.lr}, weight_decay={args.weight_decay}).")
    logging.info(f"LR Scheduler: ReduceLROnPlateau (factor={args.scheduler_factor}, patience={args.scheduler_patience}, min_lr={args.scheduler_min_lr}).")

    # --- Resume Training ---
    start_epoch, best_mIoU, patience_counter = 0, 0.0, 0
    latest_checkpoint_path = os.path.join(exp_path, 'latest_checkpoint.pth')
    best_checkpoint_path_save = os.path.join(exp_path, 'best_checkpoint.pth')

    if os.path.exists(latest_checkpoint_path):
        try:
            ckpt = torch.load(latest_checkpoint_path, map_location=device)
            net.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            if 'scheduler_state_dict' in ckpt:
                scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            start_epoch = ckpt['epoch'] + 1
            best_mIoU = ckpt.get('best_mIoU', 0.0)
            patience_counter = ckpt.get('patience_counter', 0)
            logging.info(f"Resuming training from epoch {start_epoch}. Loaded checkpoint: {latest_checkpoint_path}")
            logging.info(f"Resumed best mIoU: {best_mIoU:.4f}, patience counter: {patience_counter}")
        except Exception as e:
            logging.error(f"Could not resume training from {latest_checkpoint_path}: {e}. Starting from scratch.")
            start_epoch = 0 
            best_mIoU = 0.0
            patience_counter = 0

    # --- Dataset Loading ---
    # Get base dataset path from config
    base_dataset_path = DATASET_PATHS[args.dataset_name]
    
    # --- CORRECTED PATH HANDLING ---
    # For TSRS_RSNA datasets, we need to pass the base path, and ImageFolder should
    # look for the split subfolders (train, val) directly within that.
    # For other datasets, the same logic applies, ImageFolder will handle the splits.
    
    # The `make_dataset` function in `datasets.py` expects `root_dir` as the base path for the specific split.
    # So, if `base_dataset_path` is `/data/TSRS_RSNA-Epiphysis`, and `split='train'`,
    # we should pass `/data/TSRS_RSNA-Epiphysis/train` as the `root` to `ImageFolder`.
    # And `make_dataset` will then correctly look for `images/` and `GT/` inside that.
    
    train_data_path = os.path.join(base_dataset_path, 'train') # Corrected: root passed to ImageFolder is the specific split's directory
    val_data_path = os.path.join(base_dataset_path, 'val')   # Corrected: root passed to ImageFolder is the specific split's directory

    logging.info(f"Loading training data from: {train_data_path}")
    train_set = ImageFolder(train_data_path, args.dataset_name, args, split='train') # Pass the specific split's path
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=True, collate_fn=custom_collate_fn)

    logging.info(f"Loading validation data from: {val_data_path}")
    test_set = ImageFolder(val_data_path, args.dataset_name, args, split='val') # Pass the specific split's path
    test_loader = DataLoader(test_set, batch_size=1, num_workers=args.num_workers, shuffle=False, pin_memory=True, collate_fn=custom_collate_fn)

    logging.info(f"Number of training samples: {len(train_set)}")
    logging.info(f"Number of validation samples: {len(test_set)}")

    # --- Training Loop ---
    for epoch in range(start_epoch, args.epochs):
        net.train() # Set model to training mode
        loss_recorder = AvgMeter() # Reset loss tracker for the epoch
        
        train_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)
        
        for i, data in enumerate(train_iterator):
            if data is None: 
                logging.warning(f"Epoch {epoch+1}, Iter {i}: Skipping empty training batch due to corrupted/missing samples.")
                continue

            inputs = data['image'].to(device)
            labels = data['label'].to(device)
            
            optimizer.zero_grad(set_to_none=True) 
            
            outputs = net(inputs) 
            
            total_loss = 0
            for head_idx, pred_output in enumerate(outputs):
                current_focal_loss = focal_loss_fn(pred_output, labels.long())
                current_dice_loss = dice_loss_fn(pred_output, labels.long())
                
                combined_loss_per_head = (args.focal_loss_weight * current_focal_loss) + \
                                         (args.dice_loss_weight * current_dice_loss)
                
                total_loss += args.deep_supervision_weights[head_idx] * combined_loss_per_head
            
            total_loss.backward()
            optimizer.step()
            
            loss_recorder.update(total_loss.item(), inputs.size(0))
            train_iterator.set_postfix(loss=loss_recorder.avg)
            
        # --- Validation after each epoch ---
        current_mIoU = evaluate_model(net, test_loader, device, focal_loss_fn, dice_loss_fn, 
                                      args.deep_supervision_weights, args.focal_loss_weight, args.dice_loss_weight, mode="Validating")

        # --- Scheduler Step ---
        scheduler.step(current_mIoU)

        # --- Checkpointing ---
        if current_mIoU > best_mIoU:
            best_mIoU = current_mIoU
            patience_counter = 0 
            torch.save(net.state_dict(), best_checkpoint_path_save)
            logging.info(f"✅ New best mIoU: {best_mIoU:.4f}. Saving best model to {best_checkpoint_path_save}.")
        else:
            patience_counter += 1 
            logging.info(f"⚠️ Validation mIoU did not improve for {patience_counter} epoch(s). Best mIoU: {best_mIoU:.4f}.")

        # Save the latest checkpoint (for resuming training)
        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(), 
            'best_mIoU': best_mIoU,
            'patience_counter': patience_counter
        }, latest_checkpoint_path)
        logging.info(f"Saved latest checkpoint to {latest_checkpoint_path}")
        
        # --- Early Stopping ---
        if patience_counter >= args.patience:
            logging.info(f"Early stopping triggered: Validation mIoU did not improve for {args.patience} epochs.")
            break

    logging.info("Training finished.")
    logging.info(f"Best validation mIoU achieved: {best_mIoU:.4f}")

if __name__ == '__main__':
    main()