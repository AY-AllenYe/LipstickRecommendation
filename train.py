import jittor as jt
from utils.load_dataset import LipstickDataset, get_transforms
from utils.models import get_model
from utils.compute_weights import compute_class_weights
from jittor import nn
from jittor.dataset import DataLoader
from tqdm import trange
import sys
import datetime
from utils.logger import Logger

# jt.flags.use_cuda = 1

# 重定向标准输出
sys.stdout = Logger()

BATCH_SIZE = 68
NUM_CLASSES = 5
EPOCHS = 80

train_set = LipstickDataset(
    image_dir='datasets/TrainSet/images',
    label_file='datasets/TrainSet/labels/train.txt',
    transform=get_transforms()
)
val_set = LipstickDataset(
    image_dir='datasets/TrainSet/images',
    label_file='datasets/TrainSet/labels/val.txt',
    transform=get_transforms()
)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

model = get_model(NUM_CLASSES)
class_weights = compute_class_weights('datasets/TrainSet/labels/train.txt', NUM_CLASSES)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = nn.Adam(model.parameters(), lr=1e-4)  # resnet50 resnet18
# optimizer = nn.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

best_train_acc = 0
best_train_acc_num = 0
best_train_loss = 99999
best_train_loss_num = 0

# best_val_acc = 0
# best_val_acc_num = 0
# best_val_loss = 99999
# best_val_loss_num = 0

for epoch in trange(EPOCHS):
    model.train()
    train_total_loss, train_correct, train_total = 0, 0, 0
    for train_images, train_labels in train_loader:
        if isinstance(train_labels, (list, tuple)):
            train_labels = [l if isinstance(l, jt.Var) else jt.int32([l]) for l in train_labels]
            train_labels = jt.concat(train_labels, dim=0)

        # print("images.shape =", images.shape)
        # print("labels.shape =", labels.shape)

        train_preds = model(train_images)
        train_loss = criterion(train_preds, train_labels)
        optimizer.step(train_loss)
        train_total_loss += train_loss.item()

        train_labels_classes = jt.argmax(train_labels, dim=1)[1]
        train_pred_classes = jt.argmax(train_preds, dim=1)[0]
        # print(f"[DEBUG] pred_classes type: {type(pred_classes)}, shape: {pred_classes.shape}")
        # print(f"[DEBUG] labels type: {type(labels)}, shape: {labels.shape}")
        
        # print(train_labels)
        # print(jt.argmax(train_labels, dim=1)[1])
        # print(jt.argmax(train_preds, dim=1)[0])
        
        train_correct += (train_pred_classes == train_labels_classes).sum().item()
        train_total += train_labels.shape[0]
        # print(f'train_correct: {train_correct}; train_total: {train_total}')

    train_acc = train_correct / train_total

    if train_acc > best_train_acc:
        best_train_acc = train_acc
        best_train_acc_num = epoch
        jt.save(model.state_dict(), f"models/best_train_acc_model.pkl")
        # print(f"New best model saved at epoch {epoch}, acc={acc:.4f}")
    if train_loss < best_train_loss:
        best_train_loss = train_loss
        best_train_loss_num = epoch
        jt.save(model.state_dict(), f"models/best_train_loss_model.pkl")
        # print(f"New best model saved at epoch {epoch}, acc={acc:.4f}")
    # jt.save(model.state_dict(), f"model_save/model_{epoch}.pkl")

    # val_total_loss, val_correct, val_total = 0, 0, 0
    # for val_images, val_labels in val_loader:
    #     if isinstance(val_labels, (list, tuple)):
    #         val_labels = [l if isinstance(l, jt.Var) else jt.int32([l]) for l in val_labels]
    #         val_labels = jt.concat(val_labels, dim=0)

    #     # print("images.shape =", images.shape)
    #     # print("labels.shape =", labels.shape)

    #     val_preds = model(val_images)
    #     val_loss = criterion(val_preds, val_labels)
    #     optimizer.step(val_loss)
    #     val_total_loss += val_loss.item()

    #     val_labels_classes = jt.argmax(val_preds, dim=1)[1]
    #     val_pred_classes = jt.argmax(val_preds, dim=1)[0]
    #     # print(f"[DEBUG] pred_classes type: {type(pred_classes)}, shape: {pred_classes.shape}")
    #     # print(f"[DEBUG] labels type: {type(labels)}, shape: {labels.shape}")
        
    #     # print(val_labels)
    #     print(jt.argmax(val_labels, dim=1)[1])
    #     print(jt.argmax(val_preds, dim=1)[0])
        
    #     val_correct += (val_pred_classes == val_labels_classes).sum().item()
    #     val_total += val_labels.shape[0]
    #     print(f'val_correct: {val_correct}; val_total: {val_total}')

    # val_acc = val_correct / val_total

    # if val_acc > best_val_acc:
    #     best_val_acc = val_acc
    #     best_val_acc_num = epoch
    #     jt.save(model.state_dict(), f"models/best_val_acc_model.pkl")
    #     # print(f"New best model saved at epoch {epoch}, acc={acc:.4f}")
    # if val_loss < best_val_loss:
    #     best_val_loss = val_loss
    #     best_val_loss_num = epoch
    #     jt.save(model.state_dict(), f"models/best_val_loss_model.pkl")
    #     # print(f"New best model saved at epoch {epoch}, acc={acc:.4f}")
    # # jt.save(model.state_dict(), f"model_save/model_{epoch}.pkl")

    # print(f"Epoch {epoch}: Train Loss={train_total_loss:.4f}, Train Acc={train_acc:.4f}; Val Loss={val_total_loss:.4f}, Val Acc={val_acc:.4f}")
    print(f"Epoch {epoch}: Train Loss={train_total_loss:.4f}, Train Acc={train_acc:.4f}")
    if (epoch + 1) % 5 == 0:
        print(f"till Epoch {epoch}:")
        print(f"Epoch {best_train_acc_num} has highest train accucracy {best_train_acc}")
        print(f"Epoch {best_train_loss_num} has lowest train loss {best_train_loss}")
        # print(f"Epoch {best_val_acc_num} has highest val accucracy {best_val_acc}")
        # print(f"Epoch {best_val_loss_num} has lowest val loss {best_val_loss}")

print("ALL Done!")
print(f"Epoch {best_train_acc_num} has highest train accucracy {best_train_acc}")
print(f"Epoch {best_train_loss_num} has lowest train loss {best_train_loss}")
# print(f"Epoch {best_val_acc_num} has highest val accucracy {best_val_acc}")
# print(f"Epoch {best_val_loss_num} has lowest val loss {best_val_loss}")
