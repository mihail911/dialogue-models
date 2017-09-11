import collections
import dill as pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time

# Hack for command line invocations                                                                                                     
if __name__ == '__main__':
    import os, sys
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(root_dir)

# Use our own logger instead
try:
    from util.afs_safe_logger import Logger
except ImportError as e:
    import warnings
    warnings.warn("Could not find logger.  Using base logger instead.")
    from BaseLogger import BaseLogger as Logger
# Number of seconds in an hour
SEC_HOUR = 3600


class Stats(object):
    """
    General purpose object for recording and logging statistics/run of model run including
    accuracies and cost values. Will also be used to plot appropriate graphs.
    Note 'expName' must be full path for where to log experiment info.
    """
    def __init__(self, exp_name):
        self.logger = Logger(exp_name)
        self.startTime = time.time()
        self.timeElapsed = 0
        self.acc = collections.defaultdict(list)
        self.cost = []
        # Maintain metrics such as perplexity, etc. for various datasets
        self.metrics = \
            collections.defaultdict(lambda: collections.defaultdict(list))
        self.exp_name = exp_name


    def log(self, message):
        self.logger.Log(message)


    def reset(self):
        for _, metric_values in self.metrics.iteritems():
            metric_values.clear()
        self.totalNumEx = 0


    def record_metric(self, numEx, value, metric_name, dataset="train"):
        """
        Record arbitrary metric.
        :param numEx:
        :param acc:
        :param metric_name:
        :return:
        """
        self.timeElapsed = time.time() - self.startTime

        self.metrics[metric_name][dataset].append((numEx, value))
        self.logger.Log("Current " + dataset + " {2} after {0} examples:"
                                               " {1}".format(numEx, value, metric_name))


        if dataset == "train":
            ex = self.getExNum(metric_name, dataList=dataset)
            metric_value = self.getMetric(metric_name, dataList=dataset)
        elif dataset == "dev":
            ex = self.getExNum(metric_name, dataList=dataset)
            metric_value = self.getMetric(metric_name, dataList=dataset)


        # Pickle metric values
        with open(self.exp_name + "_metrics.pickle", "wb") as f:
            pickle.dump(self.metrics, f)

        self.plotAndSaveFig(self.exp_name + "_" + dataset +
                            "_{0}.png".format(metric_name), dataset +
                            "{0} vs. Num Examples".format(metric_name),
                            "Num Examples", dataset +
                            " {0}".format(metric_name), ex, metric_value)


    def plotAndSaveFig(self, fileName, title, xLabel, yLabel, xCoord, yCoord):
        plt.plot(xCoord, yCoord)
        plt.xlabel(xLabel)
        plt.ylabel(yLabel)
        plt.title(title)
        plt.savefig(fileName)
        plt.clf()

    def plotAndSaveFigs(self, fileName, title, xLabel, yLabel, coordList):
        for x, y in coordList:
            plt.plot(x, y)
        plt.xlabel(xLabel)
        plt.ylabel(yLabel)
        plt.title(title)
        plt.savefig(fileName)
        plt.clf()


    def getMetric(self, metric_name, dataList="train"):
        return [metric[1] for metric in self.metrics[metric_name][dataList]]


    def getExNum(self, metric_name, dataList="train"):
        """
        Get total examples in data list specified returned as a list.
        Options include "train", "dev", "cost".
        Eventually will support "test" as well.
        :param dataList:
        :return:
        """
        return [stat[0] for stat in self.metrics[metric_name][dataList]]
