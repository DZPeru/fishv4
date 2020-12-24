mkdir -p ~/.streamlit

bash -c 'echo -e "\
[general]\n\
email = \"\"\n\
" > ~/.streamlit/credentials.toml'

bash -c 'echo -e "\
[server]\n\
enableCORS = false\n\
" > ~/.streamlit/config.toml'

gdown --output ./fishv4/fish.weights --id 1_Uo_jB4ZVsBRA7I3MLisq_GZ8EQHeWr9