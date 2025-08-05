variable "IMAGE_TAGS" {
  default = "local"
}

function "tag" {
  params = [tag]
  result = "${split(",", "${tag}")}"
}

# Build geo open image
target "rusle" {
  context = "."
  dockerfile = "Dockerfile"
  tags = "${tag("${IMAGE_TAGS}")}"
}