from ssPty.RPI_sims.load_test_object import load_test_object


#######################################################################
# Generate initial conditions for randomized zone plate visible light experiment
#######################################################################

def create_initial_object(im_path, zoom=384/1024, im_size_choice='large', n_tile=1):
    # Backwards-compatible alias. The implementation lives in load_test_object()
    # so the object-loading logic has a single source of truth.
    return load_test_object(im_path, zoom=zoom, im_size_choice=im_size_choice, n_tile=n_tile)