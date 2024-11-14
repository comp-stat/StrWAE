model="StrWAE_semi_cls"
dataset="mnist"

learning_rate=0.001
learning_rate_gan=0.001
optim_beta=0.9
optim_gan_beta=0.5

epochs=500
batch_size=100
random_seed=2023
coordinate=true

classifier="./checkpoints/mnist_simple_jit.pt"

conv_layers=3
fc_layers=3
disc_layers=3
base_channels=32
hidden_size=128
disc_size=128
latent_dim=10
linear_bn=

lambda_gan=1
lambda_disc=1
lambda_sup=500
lambda_mmd=100
lambda_hsic=100

learning_rate_sup=0
lambda_attr=0

python main.py \
--model=${model} \
--dataset=${dataset} \
--learning-rate=${learning_rate} \
--learning-rate-gan=${learning_rate_gan} \
--learning-rate-sup=${learning_rate_sup} \
--optim-beta=${optim_beta} \
--optim-gan-beta=${optim_gan_beta} \
--epochs=${epochs} \
--batch-size=${batch_size} \
--random-seed=${random_seed} \
--coordinate=${coordinate} \
--classifier=${classifier} \
--conv-layers=${conv_layers} \
--fc-layers=${fc_layers} \
--disc-layers=${disc_layers} \
--base-channels=${base_channels} \
--hidden-size=${hidden_size} \
--disc-size=${disc_size} \
--latent-dim=${latent_dim} \
--linear-bn=${linear_bn} \
--lambda-gan=${lambda_gan} \
--lambda-disc=${lambda_disc} \
--lambda-sup=${lambda_sup} \
--lambda-mmd=${lambda_mmd} \
--lambda-hsic=${lambda_hsic} \
--lambda-attr=${lambda_attr}