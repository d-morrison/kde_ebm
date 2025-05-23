
def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('kde_ebm', parent_package, top_path)

    # submodules which do not have their own setup.py
    # we must manually add sub-submodules & tests
    config.add_subpackage('distributions')
    config.add_subpackage('event_order')
    config.add_subpackage('mcmc')
    config.add_subpackage('mixture_model')
    config.add_subpackage('plotting')
    config.add_subpackage('staging')

    # submodules which have their own setup.py
    config.add_subpackage('datasets')

    # add the test directory
    config.add_subpackage('tests')

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
