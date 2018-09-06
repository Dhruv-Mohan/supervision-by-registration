/*******************************/

1.download caffe_bins And put them in the pwd.
LINK: "https://drive.google.com/drive/folders/1_K75L7z_s-ujTOVUOwB8RfOGgSEtF7R1?usp=sharing"

2.Push the libs to the phone.Push caffe2 aten libs to the phone.
through adb from host.along with speed benchmark.via adb in host

3.push the init_net.pb and predict to the phone via adb.Connect to phone from docker container.

4.For python to find caffe2 in the docker image.
You’ll have to add caffe2 to python path.Caffe2 has been installed in /programs/pytorch/build/

5.Run python expos/eval_caffe.py,Running eval_caffe.py will cause the Aborted error.

/*************/
