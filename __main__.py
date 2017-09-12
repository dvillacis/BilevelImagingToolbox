"""
Allow bilevel_imaging_toolbox to be executable from a checkout or zip file
"""
import runpy

if __name__ == "__main__":
    runpy.run_module("bilevel_imaging_toolbox", run_name="__main__")
