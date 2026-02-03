mkdir -p ~/.streamlit/

echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
[browser]\n\
serverAddress = \"0.0.0.0\"\n\
" > ~/.streamlit/config.toml
