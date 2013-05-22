'''
Created on Oct 20, 2012

@author: Georgiana Dinu, Pham The Nghia
'''

def assert_bool(option, message, usage):
    if option not in (True, False):
        print message
        usage(1)        

def assert_option_not_none(option, message, usage):
    if option is None:
        print message
        usage(1)

def assert_xor_options(option1, option2, message, usage):
    if not ((option1 is None) ^ (option2 is None)):
        print message
        usage(1)
        
def config_get(section, config, option, default):        
    return config.get(section, option) if config.has_option(section, option) else default
