def rand_rotation(array: ndarray):
    rand_deg = random.uniform(-10, 10)
    return sk.transform.rotate(array, rand_deg)


def data_augmentor(data):
    dummy_df = []
    trans_img = None
    for i in range(len(data)):
        trans_img = rand_rotation(data.iloc[i,:-1].values.reshape((28,28)))
        if(random.randint(0,100) > 20):
            trans_img = sk.util.random_noise(trans_img,mode = 'gaussian', clip = False)
        if(random.randint(0,100) > 20):
            scale = random.randint(-10,10)/100 
            trans_img = transform.resize(transform.rescale(trans_img, 1 + scale,preserve_range = True), (28, 28), order = 0, preserve_range = False, anti_aliasing = False).astype('uint8')
        if(random.randint(0,100) > 20):
            trans_img = ndimage.gaussian_filter(trans_img, sigma=0.5)
        dummy_df.append(trans_img.flatten().tolist())
    aug_df = pd.DataFrame(dummy_df, columns=data.columns[:-1])
    aug_df['output'] = data['output']
    augmented_df = data.append(aug_df)
    return augmented_df