import torch
import numpy as np

def euclidean_dist(x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()
    return dist

def coreset_selection(matrix, budget: int, metric, device, random_seed=None, index=None, already_selected=None,
                    print_freq: int = 20):
    if type(matrix) == torch.Tensor:
        assert matrix.dim() == 2
    elif type(matrix) == np.ndarray:
        assert matrix.ndim == 2
        matrix = torch.from_numpy(matrix).requires_grad_(False).to(device)

    sample_num = matrix.shape[0]
    assert sample_num >= 1

    if budget < 0:
        raise ValueError("Illegal budget size.")
    elif budget > sample_num:
        budget = sample_num

    if index is not None:
        assert matrix.shape[0] == len(index)
    else:
        index = np.arange(sample_num)

    assert callable(metric)

    already_selected = np.array(already_selected)

    with torch.no_grad():
        np.random.seed(random_seed)
        if already_selected.__len__() == 0:
            select_result = np.zeros(sample_num, dtype=bool)
            # Randomly select one initial point.
            already_selected = [np.random.randint(0, sample_num)]
            budget -= 1
            select_result[already_selected] = True
        else:
            select_result = np.in1d(index, already_selected)

        num_of_already_selected = np.sum(select_result)# =1

        # Initialize a (num_of_already_selected+budget-1)*sample_num matrix storing distances of pool points from
        # each clustering center.
        dis_matrix = -1 * torch.ones([num_of_already_selected + budget - 1, sample_num], requires_grad=False).to(device)

        dis_matrix[:num_of_already_selected, ~select_result] = metric(matrix[select_result], matrix[~select_result])

        mins = torch.min(dis_matrix[:num_of_already_selected, :], dim=0).values#每个点和对应的already_selected最小值

        for i in range(budget):
            if i % print_freq == 0:
                print("| Selecting [%3d/%3d]" % (i + 1, budget))
            p = torch.argmax(mins).item()
            select_result[p] = True

            if i == budget - 1:
                break
            mins[p] = -1
            dis_matrix[num_of_already_selected + i, ~select_result] = metric(matrix[[p]], matrix[~select_result])
            mins = torch.min(mins, dis_matrix[num_of_already_selected + i])
    return index[select_result]

def main(txtName, numpyName, caseList, budget, device, seed):
    """
    txtName: The .txt contains the ordinal name list of images. For example:"TRAIN001/001.png"
    numpyName: The .np file containsthe ordinal 1-D feature maps of image (reshaped from 2-D feature map),
                which are acquired from the output of encoder. The size of it is (number of images, feature dimensions).
    caseList: The ordinal list of case name. For example: ['TRAIN001', 'TRAIN002', 'TRAIN003', 'TRAIN04'].
    budget: The number of images for core-set selection.
    device: The cuda number or cpu.
    seed: The random seed decides the initial point.
    """

    #loading txt list
    with open(txtName, mode='r') as F:
        imageList = F.readlines()
    #load image features
    already_selected = []
    matrix = np.load(numpyName)
    matrix = np.array(matrix, dtype=np.float32)
    print("number of images total:",matrix.shape[0],", feature dimension:",matrix.shape[1])

    for case in caseList:
        indexlist = []
        featurelist = []
        for i, imageName in enumerate(imageList):
            if case in imageName:
                indexlist.append(i)
                featurelist.append(matrix[i])
        x = torch.from_numpy(np.array(featurelist))
        y = torch.from_numpy(np.array(featurelist))
        dis_matrix = euclidean_dist(x, y)
        min = int(torch.argmin(torch.sum(dis_matrix, dim=0), dim=0).numpy())
        already_selected.append(indexlist[min])

    subset = coreset_selection(matrix, budget=int(np.round(budget)),
                                           metric=euclidean_dist, device=device,
                                           random_seed=seed,
                                           already_selected=np.array(already_selected))

    print("{} images has been selected as for core-set!".format(len(subset)))
    print("The index of selected images are as follows:")
    print(subset)

if __name__ == '__main__':

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    txtName = './train_fewshot_data.txt'
    numpyName = 'imageFeature.npy'
    caseList = ['TRAIN001', 'TRAIN002', 'TRAIN003']
    budget = 50
    seed=700

    main(txtName, numpyName, caseList, budget, device, seed)
