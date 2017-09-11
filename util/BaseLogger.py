import sys
from multiprocessing.dummy import Pool
from time import strftime


class Color: 
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   END = '\033[0m'


class BaseLogger:
    
    def __init__(self, outfile ='baselogger.log', sockets=[]):
        """
        A basic logging system.
        :param outfile: The file the logging output will be directed to.
        :param sockets: A list of external sockets who wish to be notified
        
        """
        self.outfile = 'baselogger.out'
        self.fh = open(outfile, 'a')
        self.sockets = sockets
        self.level_text = ['INFO', 'WARNING', 'ERROR']
        self.pool = Pool(2 if self.sockets else 1)

    def Log(self, message, level=0):
        """
        Fires off a job to write log information to file and the
        listening sockets.
        """
        timestamp = strftime("%Y-%m-%d %H:%M:%S")
        format_message = "# {0} [{1}]\t: {2}".format(timestamp,
                                                     self.level_text[level],
                                                     message)
        if level == 1:
            color = Color.YELLOW
        elif level == 2:
            color = Color.RED
        else:
            color = Color.GREEN
            
        sys.stdout.write(color + format_message + Color.END + "\n")
        self.pool.map_async(_log_fs, [(self.fh, format_message)] )
        self.pool.map_async(_log_socket, [(s, format_message) for s in
                                          self.sockets])
        
    def destroy(self):
       """
       Clean up resources.  Closes all open files and sockets.
       """
       self.Log("Destroying logging system.", level=1)
       self.pool.close()
       # Continue until all operations have finished
       self.fh.close()
       for s in self.sockets:
          s.close()
       self.pool.close()
       
       
        
def _log_fs(args):
    fh, message = args
    try:
        fh.write(message + "\n")
    except IOError:
        sys.stderr.write(Color.RED + "Filesystem error: could "
                                     "not write message :\n\t" +
                                      message + Color.END + "\n")
        
def _log_socket(args):
        """ 
        Callback to write to log and sockets.
        """
        sock, message = args
        try:
            sock.send(message)
        except:
            pass


if __name__ == '__main__':
   logger = BaseLogger()
   logger.Log("hello world!", level=0)
   logger.Log("This is a warning", level=1)
   logger.Log("This looks like a critical error", level=2)
   logger.Log("No error actually is happening.")
   logger.destroy()
