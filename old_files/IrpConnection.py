import pexpect
import time
import sys

class IPythonParallelMap(object):
    """ Class to handle the creation and management of cluster
    resources, typically on IRP, through IPython's parallel
    implementation. """

    def __init__(self, nodes, irp=True, debug=False):
        """ if SSH, Open a connection to IRP and start the IPCluster
        daemon """
        self.nodes = nodes
        self.irp = irp
        if self.irp:
            self.child = pexpect.spawn('ssh psj@irp.bic.ucsb.edu')
            if debug: self.child.logfile = sys.stdout  
            time.sleep(0.2)
            self.child.sendline('cd /home/psj/Documents/IPClusterLogs')
            self.child.sendline('ipcluster start --profile=pbs -n ' +
                                str(nodes) + ' --daemonize')
        else:
            self.child = pexpect.spawn('ipcluster start -n ' + str(nodes) + ' --daemonize')
            

    def close(self):
        """ Close the IPCluster, delete jobs, and logout of SSH """
        if self.irp: self.child.sendline('ipcluster stop --profile=pbs')
        else: self.child.sendline('ipcluster stop')
        time.sleep(0.5)
        if self.irp: self.child.sendline('qdel all')
        time.sleep(0.1)
        self.child.sendline('logout')

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.close()

    def connect_client(self):
        """ Connect the current client to the running engine """
        from IPython.parallel import Client 

        if self.irp: self.client = Client(profile='pbs')
        else: self.client = Client()

        assert len(self.client.ids) == self.nodes
        self.lview = self.client.load_balanced_view()
        self.dview = self.client.direct_view()
        

    def __call__(self, *args, **kwargs):
        """ Map function call to parallel view """
        results = self.lview.map(*args, balanced=True, **kwargs)
        return results.get()




