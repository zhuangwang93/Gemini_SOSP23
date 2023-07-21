export ETCDCTL_API=3
HOST_1=172.31.1.178
HOST_2=172.31.11.29
ENDPOINTS=$HOST_1:2379,$HOST_2:2379


etcdctl --endpoints=$ENDPOINTS endpoint status
etcdctl --endpoints=$ENDPOINTS member list

etcdctl --endpoints=$ENDPOINTS put foo "Hello World!"
etcdctl --endpoints=$ENDPOINTS get foo