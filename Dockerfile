
FROM ubuntu:20.04
ENV DEBIAN_FRONTEND=noninteractive
ARG OPENCV_VERSION=3.4.1
ARG NODEJS_MAJOR_VERSION=10
ARG WITH_CONTRIB=y

ENV OPENCV4NODEJS_DISABLE_AUTOBUILD=1

RUN apt-get update && \
apt-get install -y build-essential curl libpng-dev libx11-dev && \
apt-get install -y --no-install-recommends wget unzip git python cmake

RUN curl -sL https://deb.nodesource.com/setup_${NODEJS_MAJOR_VERSION}.x | bash && \
apt-get install -y nodejs && node -v && npm -v

RUN rm -rf /var/lib/apt/lists/* && \
mkdir opencv && \
cd opencv && \
wget https://github.com/Itseez/opencv/archive/${OPENCV_VERSION}.zip --no-check-certificate -O opencv-${OPENCV_VERSION}.zip && \
unzip opencv-${OPENCV_VERSION}.zip && \
if [ -n "$WITH_CONTRIB" ]; then \
	wget https://github.com/Itseez/opencv_contrib/archive/${OPENCV_VERSION}.zip --no-check-certificate -O opencv_contrib-${OPENCV_VERSION}.zip; \
	unzip opencv_contrib-${OPENCV_VERSION}.zip; \
fi && \
mkdir opencv-${OPENCV_VERSION}/build && \
cd opencv-${OPENCV_VERSION}/build && \
cmake_flags="-D CMAKE_BUILD_TYPE=RELEASE \
  -D BUILD_EXAMPLES=OFF \
	-D BUILD_DOCS=OFF \
	-D BUILD_TESTS=OFF \
	-D BUILD_PERF_TESTS=OFF \
	-D BUILD_JAVA=OFF \
	-D BUILD_opencv_apps=OFF \
	-D BUILD_opencv_aruco=OFF \
	-D BUILD_opencv_bgsegm=OFF \
	-D BUILD_opencv_bioinspired=OFF \
	-D BUILD_opencv_ccalib=OFF \
	-D BUILD_opencv_datasets=OFF \
	-D BUILD_opencv_dpm=OFF \
	-D BUILD_opencv_fuzzy=OFF \
	-D BUILD_opencv_hfs=OFF \
	-D BUILD_opencv_java_bindings_generator=OFF \
	-D BUILD_opencv_js=OFF \
  -D BUILD_opencv_img_hash=OFF \
  -D BUILD_opencv_line_descriptor=OFF \
  -D BUILD_opencv_optflow=OFF \
  -D BUILD_opencv_phase_unwrapping=OFF \
	-D BUILD_opencv_python3=OFF \
	-D BUILD_opencv_python_bindings_generator=OFF \
	-D BUILD_opencv_reg=OFF \
	-D BUILD_opencv_rgbd=OFF \
	-D BUILD_opencv_saliency=OFF \
	-D BUILD_opencv_shape=OFF \
	-D BUILD_opencv_stereo=OFF \
	-D BUILD_opencv_stitching=OFF \
	-D BUILD_opencv_structured_light=OFF \
	-D BUILD_opencv_superres=OFF \
	-D BUILD_opencv_surface_matching=OFF \
	-D BUILD_opencv_ts=OFF \
	-D BUILD_opencv_xobjdetect=OFF \
	-D BUILD_opencv_xphoto=OFF \
	-D CMAKE_INSTALL_PREFIX=/usr/local" && \
if [ -n "$WITH_CONTRIB" ]; then \
  cmake_flags="$cmake_flags -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-${OPENCV_VERSION}/modules"; \
fi && \
echo $cmake_flags && \
cmake $cmake_flags .. && \
make -j $(nproc) && \
make install && \
sh -c 'echo "/usr/local/lib" > /etc/ld.so.conf.d/opencv.conf' && \
ldconfig && \
cd ../../../ && \
rm -rf opencv && \
npm install -g opencv4nodejs --unsafe-perm

# RUN npm install -g face-recognition --unsafe-perm

ENV NODE_PATH=/usr/lib/node_modules
