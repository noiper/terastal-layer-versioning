import torch
from tqdm import tqdm
from .train_helper import partial_train, get_partial_model
from torch.nn.functional import mse_loss
import torch.nn

class LayerVersionTrainer:
    def __init__(self, 
                 model,
                 model_name,
                 teacher_model,
                 train_layers,
                 train_loader,
                 val_loader,
                 optimizer,
                 criterion,
                 num_epochs,
                 device,
                 KD=False):
        self.model = model
        self.model_name = model_name
        self.teacher_model = teacher_model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.num_epochs = num_epochs
        self.device = device
        self.train_layers = train_layers
        self.KD = KD
        self.epoch = 0

    def _train_epoch_KD(self):
        self.model.to(self.device)
        self.teacher_model.to(self.device)
        partial_train(self.model, self.train_layers)
        student = get_partial_model(self.model, self.train_layers[-1])
        teacher = get_partial_model(self.teacher_model, self.train_layers[-1])

        running_loss = 0.0
        correct = 0
        correct5 = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f'Training Epoch {self.epoch}')
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            captured = student(data, stop_early=False, detach=False, no_grad=False)
            interm = captured["interm"]
            out = captured["out"]
            teacher_interm_output = teacher(data, stop_early=True, detach=True, no_grad=True)["interm"]
            loss = self.criterion(interm, teacher_interm_output)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            
            total += target.size(0)
            if out is not None:
                _, predicted = out.max(1)
                correct += predicted.eq(target).sum().item()
            
                # top-5
                _, pred5 = out.topk(5, dim=1)
                correct5 += pred5.eq(target.view(-1, 1)).sum().item()

            # Update progress bar
            avg_loss = running_loss / (batch_idx + 1)
            top1_accuracy = 100. * correct / total
            top5_accuracy = 100. * correct5 / total

            pbar.set_postfix({
                'Loss': f'{avg_loss:.4f}',
                'top1 Acc': f'{top1_accuracy:.2f}%',
                'top5 Acc': f'{top5_accuracy:.2f}%'
            })


        return running_loss / len(self.train_loader), 100. * correct / total, 100. * correct5 / total

    def _train_epoch(self):
        partial_train(self.model, self.train_layers)
        running_loss = 0.0
        correct = 0
        correct5 = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f'Training Epoch {self.epoch}')
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # top-5
            _, pred5 = output.topk(5, dim=1)
            correct5 += pred5.eq(target.view(-1, 1)).sum().item()

            # Update progress bar
            avg_loss = running_loss / (batch_idx + 1)
            top1_accuracy = 100. * correct / total
            top5_accuracy = 100. * correct5 / total

            pbar.set_postfix({
                'Loss': f'{avg_loss:.4f}',
                'top1 Acc': f'{top1_accuracy:.2f}%',
                'top5 Acc': f'{top5_accuracy:.2f}%'
            })

        return running_loss / len(self.train_loader), 100. * correct / total, 100. * correct5 / total

    def validate(self):
        """Validate the model"""
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        correct5 = 0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Validation Epoch {self.epoch}')
            for data, target in pbar:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                val_loss += self.criterion(output, target).item()

                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

                # top-5
                _, pred5 = output.topk(5, dim=1)
                correct5 += pred5.eq(target.view(-1, 1)).sum().item()

                # Update progress bar
                accuracy = 100. * correct / total
                pbar.set_postfix({'Acc': f'{accuracy:.2f}%'})

        return val_loss / len(self.val_loader), 100. * correct / total, 100. * correct5 / total

    def train(self):
        best_val_top1_acc = 0.0
        best_val_top5_acc = 0.0
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_top1_accuracies = []
        val_top5_accuracies = []

        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            print("-" * 60)
            
            # Training
            if self.KD:
                train_loss, train_top1_acc, train_top5_acc = self._train_epoch_KD()
            else:
                train_loss, train_top1_acc, train_top5_acc = self._train_epoch()

            # Validation
            val_loss, val_top1_acc, val_top5_acc = self.validate()

            # Store metrics
            train_losses.append(train_loss)
            train_accuracies.append(train_top1_acc)
            val_losses.append(val_loss)
            val_top1_accuracies.append(val_top1_acc)
            val_top5_accuracies.append(val_top5_acc)

            print(f"Results - Train Loss: {train_loss:.4f}, Train Acc: {train_top1_acc:.2f}%, Train Top-5 Acc: {train_top5_acc:.2f}%")
            print(f"Results - Val Loss: {val_loss:.4f}, Val Acc: {val_top1_acc:.2f}%, Val Top-5 Acc: {val_top5_acc:.2f}%")

            # Save best model
            if val_top1_acc > best_val_top1_acc:
                best_val_top1_acc = val_top1_acc
                save_path = f'{self.model_name}_best.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_top1_acc': val_top1_acc,
                    'val_top5_acc': val_top5_acc,
                    'train_layers': self.train_layers,
                }, save_path)
                print(f"✅ New best model saved! Val Acc: {val_top1_acc:.2f}%")

        print(f"\n🎉 Training completed! Best validation accuracy: {best_val_top1_acc:.2f}%, Best validation Top-5 accuracy: {best_val_top5_acc:.2f}%")
        return {
            'train_losses': train_losses,
            'train_accuracies': train_accuracies,
            'val_losses': val_losses,
            'val_top1_accuracies': val_top1_accuracies,
            'val_top5_accuracies': val_top5_accuracies,
            'best_val_top1_acc': best_val_top1_acc,
            'best_val_top5_acc': best_val_top5_acc
        }