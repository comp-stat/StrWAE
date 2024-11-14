model="StrWAE_embedder"
data_dir="./data"
dataset="vggface2"

learning_rate=0.001
learning_rate_gan=0.001

epochs=300
batch_size=400
random_seed=2023
coordinate=""
classifier=""

enc_block_layers=4
dec_block_layers=4
enc_scaling_steps=1
dec_scaling_steps=1
enc_conv_steps=1
dec_conv_steps=1
enc_skip=""
dec_skip=true
enc_fc_layers=1
dec_fc_layers=1
disc_layers=5
base_channels=128
hidden_size=32
disc_size=256
latent_dim=32
linear_bn=""

lambda_gan=100
lambda_disc=1
lambda_hsic=100
lambda_attr=0

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
--enc-block-layers=${enc_block_layers} \
--dec-block-layers=${dec_block_layers} \
--enc-scaling-steps=${enc_scaling_steps} \
--dec-scaling-steps=${dec_scaling_steps} \
--enc-conv-steps=${enc_conv_steps} \
--dec-conv-steps=${dec_conv_steps} \
--enc-skip=${enc_skip} \
--dec-skip=${dec_skip} \
--enc-fc-layers=${enc_fc_layers} \
--dec-fc-layers=${dec_fc_layers} \
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