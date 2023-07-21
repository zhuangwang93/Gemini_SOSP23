import boto3
import os


class ASGAgent():
    def __init__(self):
        self.asg = boto3.client('autoscaling')
        self.ec2 = boto3.client('ec2')

    
    def get_instances_ids(self):
        response = self.asg.describe_auto_scaling_instances()
        instances_info = response['AutoScalingInstances']
        instances_ids_list = []
        for instance_info in instances_info:
            instance_id = instance_info['InstanceId']
            instances_ids_list.append(instance_id)
        return instances_ids_list


    def get_instances_info(self, instance_state=None):
        instances_ids = self.get_instances_ids()
        # batch query doesn't return the complete information
        instances_info = self.ec2.describe_instances(InstanceIds=instances_ids)['Reservations']
        self.instances_info = {}
        for instance_id in instances_ids:
            instances_info = self.ec2.describe_instances(InstanceIds=[instance_id])['Reservations'][0]['Instances'][0]
            try:
                ip = instances_info['PrivateIpAddress']
                public_ip = instances_info['PublicIpAddress']
                instance_id = instances_info['InstanceId']
                state = instances_info['State']['Name']
                if instance_state is None or instance_state == state:
                    self.instances_info[ip] = (public_ip, instance_id, state)
            except:
                pass
        return self.instances_info


    def get_instances_ips(self, instances_info=None, instance_state=None):
        if instances_info is None:
            instances_info = self.get_instances_info(instance_state)
            
        self.ips = []
        for ip, _ in instances_info.items():
            self.ips.append(ip)
        return self.ips


    def terminate_instance(self, instance_id):
        response = self.asg.terminate_instance_in_auto_scaling_group(
            InstanceId=instance_id,
            ShouldDecrementDesiredCapacity=False)
        print(response)


    def generate_hostfile(self, filename, ips, gpus_per_instance):
        if not os.path.isfile(filename):
            self.write_hostfile(filename, ips, gpus_per_instance)
        else:
            self.update_hostfile(filename, ips, gpus_per_instance)

    
    def write_hostfile(self, filename, ips_list, gpus_per_instance):
        with open(filename, 'w') as f:
            for ip in ips_list:
                f.write(f"{ip} slots={gpus_per_instance}\n")


    def diff_ips(self, ips_list_last, ips_list):
        assert(len(ips_list_last) == len(ips_list))
        new_ips = list(set(ips_list) - set(ips_list_last))
        del_ips = list(set(ips_list_last) - set(ips_list))

        index_list = []
        for i in range(len(del_ips)):
            index = ips_list_last.index(del_ips[i])
            index_list.append(index)
            ips_list_last[index] = new_ips[i]

        return ips_list_last, new_ips, del_ips, index_list


    def update_hostfile(self, filename, ips_list, gpus_per_instance):
        ips = []
        with open(filename, 'r') as f:
            for line in f:
                ip = line.strip().split(" ")[0]
                ips.append(ip)

        ips, new_ips, del_ips, index_list = self.diff_ips(ips, ips_list)
        self.write_hostfile(filename, ips, gpus_per_instance)



if __name__ == "__main__":
    ASG_agent = ASGAgent()
    instances = ASG_agent.get_instances_info()
    print(instances)
    ips = ASG_agent.get_instances_ips(instances)
    print(ips)