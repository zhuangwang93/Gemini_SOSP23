# python-etcd
import etcd
import ast

class etcdAPIs():
    def __init__(self, hosts):
        self.client = etcd.Client(host=hosts, allow_reconnect=True)
        self.lock = etcd.Lock(self.client, "etcd_lock")


    def write(self, key, value):       
        self.client.write(key, value)

    
    def read(self, key, dict_type=True):
        try:
            value = self.client.read(key).value
        except:
            return ""

        if not dict_type:
            return value

        if value == "":
            return ""
        else:
            return ast.literal_eval(value)


    def read_folder(self, key, dict_type=True):
        r = None
        try:
            r = self.client.read(key, recursive=True)
        except:
            return {}
    
        ret = {}
        for child in r.children:
            child_key = child.key.replace(key+"/", "")
            value = child.value
            if not dict_type or value == "":
                ret[child_key] = value
            else:
                ret[child_key] = ast.literal_eval(value)
        return ret


    def delete(self, key):
        try:
            self.client.delete(key)
        except:
            pass


    def delete_folder(self, key):
        try:
            self.client.delete(key, recursive=True)
        except:
            pass



if __name__ == "__main__":
    hosts = (("172.31.0.182", 2379),)
    etcd_client = etcdAPIs(hosts)
    key = "/agents/tests"

    etcd_client.write(key+"/1", {1:1, 2:2})

    import time
    time.sleep(1)
    key = "/vote"
    print(etcd_client.read_folder(key, dict_type=True))