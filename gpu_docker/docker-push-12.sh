NO_TAG="ghcr.io/${1}/pyafq_gpu_cuda_12"
NO_TAG="$(echo "${NO_TAG}" | tr -d '[:space:]')"

docker push --all-tags $NO_TAG
