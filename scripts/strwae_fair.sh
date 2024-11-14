model="StrWAE_stack_cond"
data_dir="./data/adult"
dataset="adult"

learning_rate=0.001
learning_rate_gan=0.001

epochs=1200
batch_size=100
random_seed=1994
coordinate=""
classifier=""

conv_layers=0
fc_layers=2
disc_layers=4
base_channels=0
hidden_size=100
disc_size=128
latent_dim=32
linear_bn=""

lambda_disc=1
lambda_gan=2
lambda_hsic=10
lambda_attr=1000000

python main.py \
--model=${model} \
--data-dir=${data_dir} \
--dataset=${dataset} \
--learning-rate=${learning_rate} \
--learning-rate-gan=${learning_rate_gan} \
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
--lambda-hsic=${lambda_hsic} \
--lambda-attr=${lambda_attr}
