def pca(
    mat,
    pca=None,
    whitening=False,
    pca_comps=10,
    verbose=False,
):
    mat_transform = mat.reshape(
        mat.shape[0], mat.shape[1] * mat.shape[2] * mat.shape[3] * mat.shape[4]
    )
    if pca == None:
        pca = PCA(n_components=pca_comps, whiten=whitening)
        mat_transform = pca.fit_transform(mat_transform)

    else:
        mat_transform = pca.transform(mat_transform)

    cum_explained_var = []
    for i in range(0, len(pca.explained_variance_ratio_)):
        if i == 0:
            cum_explained_var.append(pca.explained_variance_ratio_[0])
        else:
            cum_explained_var.append(
                pca.explained_variance_ratio_[i] + cum_explained_var[i - 1]
            )

    pc0 = pca.components_[0]
    # print(np.shape(pc0))
    pc0 = pc0.reshape(1, mat.shape[1], mat.shape[2], mat.shape[3], mat.shape[4])

    if verbose:
        print("individual explained vars: \n" + str(pca.explained_variance_ratio_))
        print("cumulative explained vars ratio: \n" + str(cum_explained_var))

    return mat_transform, pca


def unwrap_pca(mat, pca, shape):
    """
    Take as input a matrix that has been transformed by PCA and return the original matrix
    """
    mat = pca.inverse_transform(mat)
    mat = mat.reshape(len(mat), shape[1], shape[2], shape[3], shape[4])
    return mat
