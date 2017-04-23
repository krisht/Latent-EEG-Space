#!/usr/bin/env python

# revision history:
#  20160531 (JM): initial version
#
# usage:
#  nedc_print_labels.py *.lbl
#
# This script is a self contained and portable way to read lab files. It will
# then print the info to the console.
#------------------------------------------------------------------------------

# import system modules
#
import argparse
import os
import sys

#------------------------------------------------------------------------------
#
# global variables are listed here
#
#------------------------------------------------------------------------------

# define the help file and usage message
#
NEDC_HELP_MESSAGE = ("name: nedc_print_labels\n"
                     "synopsis: nedc_print_labels files\n"
                     "descr: prints out annotations of an lbl file in a simple format\n"
                     "example: nedc_print_labels file.lbl\n"
                     "\n" 
                     "options: none\n"
                     "\n"
                     "arguments: label (.lbl) files\n"
                     "\n"
                     "output format: start time, end time, channel label, label")

NEDC_USAGE_MESSAGE = "Usage: nedc_print_labels file.lbl"

# define the defaults
#
NEDC_DEF_NUM_CH = 22
NEDC_DEF_NUM_SYMB = 6

# define the default event labels
#
NEDC_DEF_LABELS = {0: '(null)', 1: 'spsw', 2: 'gped', 3: 'pled', 
                   4: 'eyem', 5: 'artf', 6: 'bckg'}

#------------------------------------------------------------------------------
#
# classes are listed here
#
#------------------------------------------------------------------------------

# class: CommandLineParser
#
# This class inherits the ArgumentParser object.
# This class is used to edit the format_help method found in the argparse
# module. It defines the help method for this specific tool.
#
class CommandLineParser(argparse.ArgumentParser):
    
    # method: Constructor
    #
    # arguments:
    #  -usage_file: a short explanation of the command that is printed when
    #               it is run without arguments.
    #  -help_file: a full explanation of the command that is printed when
    #              it is run with the -help argument.
    #
    # return: none
    #
    def __init__(self):
        argparse.ArgumentParser.__init__(self)

    # method: print_usage
    #
    # arguments: none
    #
    # This method is used to print the usage message
    #
    def print_usage(self, void):
        print NEDC_USAGE_MESSAGE

    # method: print_help
    #
    # arguments: none
    #
    # return: none
    #
    # This method is used to print the help message
    #
    def print_help(self):
        print NEDC_HELP_MESSAGE

#------------------------------------------------------------------------------
#
# the main program starts here
#
#------------------------------------------------------------------------------

# method: main
#
# arguments:
#  argv: for argument parsing
#
# return: none
#
# This method is the main function.
#
def main(argv):

    # create a command line parser
    #
    parser = CommandLineParser()
    parser.add_argument("lbl_file", type = str, nargs = 1)
    parser.add_argument("-help", action="help")

    # parse the command line
    #
    args = parser.parse_args()

    # check if the lab file exists
    #
    if not os.path.exists(args.lbl_file[0]):
        print "%s: lbl file does not exist." %sys.argv[0]
        exit(-1)

    # define a variable to watch for where the program is in the lbl file.
    # also, create lists for channel_mapping and symbol_mapping of the lbl.
    #
    mark = False
    channels_mapping = [0]*NEDC_DEF_NUM_CH
    symbol_mapping = [0]*NEDC_DEF_NUM_SYMB

    # open the .lbl file.
    #
    with open(args.lbl_file[0], "r") as fp:
        for line in fp.readlines():
            # process the channel mappings
            #
            parts0 = line.split(" = ")
            if parts0[0] == "channels":
                parts1 = parts0[1].strip().strip(";").strip("{").strip("}").split(",")

                for part in parts1:
                    part = part.split(": ")
                    channels_mapping[int(part[0])] = part[1].strip("}").strip(";").strip("/n")
                #
                # end of for

            # process symbols
            #
            if parts0[0] == "levels_symbol[0]":
                parts1 = parts0[1].strip().strip(";").strip("}").strip("{").split(",")
                for part in parts1:
                    part = part.split(": ")
                    symbol_mapping[int(part[0])] = part[1].strip("/n").strip(";").strip("}")

            # if in the sublevel 2 section of the lbl, set the mark to true.
            #
            if "level:0 sublevel:2" in line:
                mark = True
            
            # if mark is true, process the labels.
            #
            if mark == True:
                if parts0[0] == "label":
                    
                    # process timing and channel label
                    #
                    parts1 = parts0[1].strip().strip("{").strip("}").strip(";")
                    parts1 = parts1.split(" [")
                    parts2 = parts1[0].strip().strip(",").split(", ")
                    start = float(parts2[2])
                    end = float(parts2[3])
                    
                    # try to make the number defining 'channel' an int.
                    # this is used because the term 'all' is used in some
                    # sublevels to define that all channels are being used.
                    #
                    try:
                        channel = int(parts2[4])
                    except:
                        channel = parts2[4].strip()
                        pass
                    
                    # process probablility list for events
                    #
                    parts2 = parts1[1].strip("}").strip("]").split(",")
                    for i in range(len(parts2)):
                        parts2[i] = float(parts2[i])
                    #
                    # end of for
                        
                    max_num = max(parts2)
                    label = symbol_mapping[parts2.index(max_num)]

                    # print results
                    #
                    if type(channel) == int:
                        print('%.4f, %.4f, "%s", %s'
                              % (start, end, channels_mapping[channel], label))
                        
                    else:
                        print('%.4f, %.4f, "%s", %s' 
                              % (start, end, channel, label))

            # set mark to false when done processing sublevel 2.
            #
            if "level:1 sublevel:0" in line:
                mark = False
    #
    # end of with

#
# exit gracefully
                
# begin gracefully
#
if __name__ == "__main__":
    main(sys.argv[0:])

#
# end of file