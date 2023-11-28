import torch
import gol, utils
from dataloader import GraphData
from model import LightGCN


def test_one_batch(X):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = utils.getLabel(groundTrue, sorted_items)
    recall_, pre_ = utils.RecallPrecision_ATk(groundTrue, r, gol.topk)
    ndcg_ = utils.NDCGatK_r(groundTrue,r, gol.topk)
    return pre_, recall_, ndcg_

def test_model(recModel: LightGCN, dataset: GraphData):
    recModel.eval()
    testDict: dict = dataset.testDict
    all_pre, all_recall, all_ndcg = 0., 0., 0.
    with torch.no_grad():
        users = list(testDict.keys())
        users_list, rating_list, groundTrue_list = [], [], []
        total_batch = len(users) // gol.TEST_BATCH_SZ + 1
        for batch_users in utils.minibatch(users, batch_size=gol.TEST_BATCH_SZ):
            allPos = dataset.getUserPosItems(batch_users)
            groundTrue = [testDict[u] for u in batch_users]
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(gol.device)

            rating = recModel.getUsersRating(batch_users_gpu)
            exclude_index, exclude_items = [], []
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            rating[exclude_index, exclude_items] = -(1<<10)
            _, rating_K = torch.topk(rating, k=gol.topk)

            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(groundTrue)

        assert total_batch == len(users_list)
        X = zip(rating_list, groundTrue_list)
        for x in X:
            pre_, recall_, ndcg_ = test_one_batch(x)
            all_pre += pre_
            all_recall += recall_
            all_ndcg += ndcg_

        num_users = len(users)
        all_recall /= num_users
        all_pre /= num_users
        all_ndcg /= num_users
        return all_recall, all_ndcg, all_pre

def bpr_train(recModel: LightGCN, dataset: GraphData):
    opt = torch.optim.Adam(recModel.parameters(), lr=gol.conf['lr'])
    best_ndcg, best_recall, best_epoch = 0., 0., 0.
    for epoch in range(gol.EPOCH + 1):
        S = utils.UniformSample(dataset).to(gol.device)
        allUsers, posItems, negItems = utils.shuffle(S[:, 0], S[:, 1], S[:, 2])
        total_batch = len(allUsers) // gol.BATCH_SZ + 1
        ave_loss, ave_bpr, ave_l2 = 0., 0., 0.
        recModel.train()

        for batch_users, batch_pos, batch_neg in utils.minibatch(allUsers, posItems, negItems, batch_size=gol.BATCH_SZ):
            bpr, l2 = recModel.getLoss(batch_users, batch_pos, batch_neg)
            loss = bpr + l2 * gol.conf['decay']

            opt.zero_grad()
            loss.backward()
            opt.step()

            ave_loss += loss.item()
            ave_bpr += bpr.item()
            ave_l2 += l2.item()

        ave_loss /= total_batch
        ave_bpr /= total_batch
        ave_l2 /= total_batch

        if epoch % 5 == 0:
            recall, ndcg, pre = test_model(recModel, dataset)
            if best_ndcg < ndcg:
                best_ndcg, best_recall, best_epoch = ndcg, recall, epoch
            if epoch % 10 == 0:
                gol.pLog(f'Epoch {epoch} / {gol.EPOCH}, loss = {ave_loss:.5f}')
                gol.pLog(f'BPR: {ave_bpr:.4f}, L2: {ave_l2:.4f},')
                gol.pLog(f'Recall: {recall:.5f}, NDCG: {ndcg:.5f}, Precesion: {pre:.5f}')
                gol.pLog(f'Current Best Recall: {best_recall:.5f}, NDCG: {best_ndcg:.5f} at Epoch {best_epoch}\n')

    return best_ndcg, best_recall, best_epoch


if __name__ == '__main__':
    ds = GraphData(path=gol.DATA_PATH, data=gol.dataset)
    model = LightGCN(ds).to(gol.device)

    gol.pLog('Start Training\n')
    test_ndcg, test_recall, test_epoch = bpr_train(model, ds)
    gol.pLog(f'Training on {gol.dataset.upper()} finished, NDCG: {test_ndcg}, Recall: {test_recall} at epoch {test_epoch}')
