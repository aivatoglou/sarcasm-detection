import torch
from sklearn.metrics import f1_score, classification_report


def train(model, iterator, optimizer, criterion, device):

    epoch_loss = 0
    model.train()

    for batch in iterator:

        batch = tuple(b.to(device) for b in batch)

        optimizer.zero_grad()

        predictions = model(batch[0], batch[1]).squeeze(1)

        b_loss = criterion(predictions, batch[2].to(torch.int64))

        # Clip gradients
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

        b_loss.backward()
        optimizer.step()

        epoch_loss += b_loss.item()

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion, device, print_report):

    epoch_loss = 0
    epoch_score = 0

    epoch_preds = []
    epoch_label = []

    model.eval()
    with torch.no_grad():

        for batch in iterator:

            batch = tuple(b.to(device) for b in batch)

            predictions = model(batch[0], batch[1]).squeeze(1)

            # Aggregate batch F1-macro scores
            f_pred = predictions.argmax(1)
            b_score = f1_score(
                batch[2].cpu().detach().numpy(),
                f_pred.cpu().detach().numpy(),
                average="macro",
            )

            b_loss = criterion(predictions, batch[2].to(torch.int64))
            epoch_loss += b_loss.item()
            epoch_score += b_score

            if print_report:
                epoch_preds.append(f_pred)
                epoch_label.append(batch[2].cpu().detach().numpy())

    if print_report:
        epoch_preds = [w.tolist() for ep in epoch_preds for w in ep]
        epoch_label = [w for ep in epoch_label for w in ep]

        print(classification_report(epoch_label, epoch_preds))

    return epoch_loss / len(iterator), epoch_score / len(iterator)
