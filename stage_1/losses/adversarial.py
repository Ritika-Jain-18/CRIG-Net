import torch


def wgan_generator_loss(fake_scores):
    return -fake_scores.mean()


def wgan_discriminator_loss(real_scores, fake_scores):
    return fake_scores.mean() - real_scores.mean()
