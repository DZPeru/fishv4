mkdir -p ~/.streamlit

bash -c 'echo -e "\
[general]\n\
email = \"\"\n\
" > ~/.streamlit/credentials.toml'

bash -c 'echo -e "\
[server]\n\
enableCORS = false\n\
" > ~/.streamlit/config.toml'

gdown --output ./fishv4/fish.weights --id 1vosRfnj3DBkZYFrzJep5_D1DV-0b6Tpv