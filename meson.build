project(
  'minpack_numba',
  'fortran',
  version: '0.1.0',
  default_options: [
    'buildtype=release',
  ]
  )
add_languages('c', native: false)

py3 = import('python').find_installation(pure: false)

subdir('minpack_numba')

minpack_header = files(
  'minpack/include/minpack.h',
)

shared_library('minpack',
    [
    'minpack/src/minpack.f90', 
    'minpack/src/minpack_capi.f90'
    ],
    install: true,
    install_dir: py3.get_install_dir() / 'minpack_numba',
)

install_headers(minpack_header)
