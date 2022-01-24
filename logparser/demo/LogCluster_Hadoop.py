#!/usr/bin/env python
import sys
sys.path.append('../logparser/LogCluster')
import LogCluster
from os import walk

filesDir = "../../loglizer/data/to_process"
filesDirDump = filesDir + "_output"
for (dirpath, dirnames, filenames) in walk(filesDir):
    print(filenames)
    for filename in filenames:
        print("dirpath:",dirpath)
        print("filename:",filename)
        if "abnormal_label" not in filename and "BGL" not in filename:
            print("=====")
            input_dir  = dirpath # The input directory of log file
            output_dir = str.replace(dirpath, filesDir, filesDir+"_output") # The output directory of parsing results
            log_file   = filename # The input log file name
            log_format = '<Date> <Time>,<PID> <Level> <Process> <Component>: <Content>' # HDFS log format
            rsupport   = 10 # The minimum threshold of relative support, 10 denotes 10%
            regex      = [] # Regular expression list for optional preprocessing (default: [])
            print(input_dir)
            print(output_dir)
            print("\n")
            parser = LogCluster.LogParser(input_dir, log_format, output_dir, rsupport=rsupport)
            #parser = LogCluster.LogParser(input_dir, log_format, output_dir, rsupport=rsupport)
            parser.parse(log_file)

# input_dir  = '../logs/HDFS/' # The input directory of log file
# output_dir = 'LogCluster_result/' # The output directory of parsing results
# log_file   = 'HDFS_2k.log' # The input log file name
# log_format = '<Date> <Time> <Pid> <Level> <Component>: <Content>' # HDFS log format
# rsupport   = 10 # The minimum threshold of relative support, 10 denotes 10%
# regex      = [] # Regular expression list for optional preprocessing (default: [])

# input_dir  = '../logs/Hadoop/' # The input directory of log file
# output_dir = 'LogCluster_result2/' # The output directory of parsing results
# log_file   = 'Hadoop_2k.log' # The input log file name
# log_format = '<Date> <Time>,<PID> <Level> <Process> <Component>: <Content>' # HDFS log format
# rsupport   = 10 # The minimum threshold of relative support, 10 denotes 10%
# regex      = [] # Regular expression list for optional preprocessing (default: [])



#parser = LogCluster.LogParser(input_dir, log_format, output_dir, rsupport=rsupport)
# parser = LogCluster.LogParser(input_dir, log_format, output_dir, rsupport=rsupport)
# parser.parse(log_file)
