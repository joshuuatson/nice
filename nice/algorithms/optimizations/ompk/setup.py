# NICE
# Copyright (C) 2017 - Authors of NICE
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# You can be released from the requirements of the license by purchasing a
# commercial license. Buying such a license is mandatory as soon as you
# develop commercial activities as mentioned in the GNU Affero General Public
# License version 3 without disclosing the source code of your own applications.
#
import os
from distutils.core import setup, Extension

from numpy import get_include

conda_env = os.getenv('CONDA_PREFIX', None)

include = [get_include(), '.']

if conda_env is not None:
      include = [f'{conda_env}/include'] + include

module1 = Extension('ompk',
                    sources=['ompk.c', 'komplexity.c'],
                    extra_compile_args=['-lz', '-fopenmp', '-std=c99', '-O3'],
                    extra_link_args=['-lz', '-fopenmp'],
                    include_dirs=include)

setup(name='OmpkBindings',
      version='1.0',
      description='This is the OmpK bindings to python interface',
      ext_modules=[module1])
