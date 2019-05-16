from imgaug import augmenters as iaa


def default():
    st = lambda aug, p=0.15: iaa.Sometimes(p, aug)  # noqa

    seq = iaa.Sequential([
        st(iaa.Superpixels(p_replace=0.2, n_segments=(64, 256))),
        st(iaa.CropAndPad(percent=(-0.25, 0.25))),
        st(iaa.GaussianBlur(sigma=(0.0, 1.5))),
        st(iaa.Grayscale(alpha=(0.1, 0.3))),
        st(iaa.PiecewiseAffine(scale=(0.01, 0.05))),
        st(iaa.Add((-40, 40))),
        st(iaa.AdditiveGaussianNoise(loc=0., scale=(0.1, 10)), 0.4)
    ])
    return seq


imgaug_seqs_dict = {
    'default': default()
}
