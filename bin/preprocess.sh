docker run --rm -it \
    -v $PWD/:/root/workdir/ \
    -v $HOME/.ssh/:/root/.ssh \
    -v $HOME/.config/:/root/.config \
    -v $HOME/.netrc/:/root/.netrc \
    -v $HOME/.cache/:/root/.cache \
    --ipc=host \
    kaggle:cuda11 \
    python ./scripts/make_fold.py

