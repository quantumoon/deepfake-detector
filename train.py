import os
from data import download_data, build_dataloaders
from model import LitDeepfakeDetector
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor


def main():
    # 0. Download data & set up variables
    if not os.path.isdir("train"):
        print("Training data hasn't been found. Trying to download.")
        train_data_url = "https://disk.360.yandex.ru/d/SKW4m1c-2l76SQ"
        download_data(train_data_url, name="train")
        
    if not os.path.isdir("test"):
        print("Test data hasn't been found. Trying to download.")
        test_data_url = "https://disk.360.yandex.ru/d/rQQqIAI_YGpjJA"
        download_data(test_data_url, name="test")
    print("All data collected")
    
    CSV_TRAIN = 'train.csv'
    ROOT_TRAIN = './train'
    CSV_TEST = 'test.csv'
    ROOT_TEST = './test'
    BATCH_SIZE = 64
    EPOCH = 5
    LOG_STEPS = 50
    CHECK_STEPS = 20
    

    # 1. Get dataloaders
    train_loader, val_loader, test_loader = build_dataloaders(csv_file_train=CSV_TRAIN,
                                                              root_dir_train=ROOT_TRAIN,
                                                              csv_file_test=CSV_TEST,
                                                              root_dir_test=ROOT_TEST,
                                                              batch_size=BATCH_SIZE)

    # 2. Set up the model & compute GFLOPs
    model = LitDeepfakeDetector(num_types=NUM_TYPES)
    g = model.flops()
    print(f"GFLOPs per 1 image = {g:.2f}" if g else "Framework thop is missing to compute GFLOPs per 1 image")

    # 3. Set up loggers and trainer
    logger = TensorBoardLogger(save_dir="tb_logs", name="deepfake_exp")
    
    ckpt_cb = ModelCheckpoint(monitor="val/roc_auc",
                              mode="max",
                              filename="best-auc-{epoch:02d}-{val/roc_auc:.4f}",
                              save_top_k=1,
                              verbose=True)
    
    lr_cb = LearningRateMonitor(logging_interval="epoch")
    
    trainer = pl.Trainer(accelerator="gpu" if torch.cuda.is_available() else "cpu",
                         devices=1,
                         max_epochs=EPOCH,
                         logger=logger,
                         callbacks=[ckpt_cb, lr_cb],
                         log_every_n_steps=LOG_STEPS,
                         num_sanity_val_steps=CHECK_STEPS)

    # 4. Train the model
    trainer.fit(model, train_loader, val_loader)

    # 5. Obtain predictions from the model
    preds = trainer.predict(model, test_loader)
    probs = torch.cat(preds).cpu().numpy()

    # 6. Save predictions for submittion
    df = pd.read_csv(ROOT_TEST + '/' + CSV_TEST)
    df["prob"] = probs
    df.to_csv('submission.csv', index=False)

    print('Predictions are saved as submission.csv')

if __name__ == '__main__':
    main()