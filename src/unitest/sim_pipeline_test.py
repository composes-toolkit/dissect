import numpy as np

import pipelines.compute_similarities as sim_pipeline
from pipelines import build_core_space as bcs

import pytest


def read_number_list(file_name, column):
    result = []
    with open(file_name) as f:
        for line in f:
            line = line.strip()
            if line:
                elements = line.split()
                if column >= len(elements):
                    raise ValueError("Expected line to have at least %d elements: %s" % (column + 1, line.strip()))
                result.append(float(elements[column]))
                    
    return result 

    return result

    def setUp(self):
        self.dir_ = data_dir
        self.log_dir = toolkit_dir + "/log/"
        self.cos = np.array([1.0, 0.585984772383, 1.0, 0.585984772383])
        self.dot_prod = np.array([28.0, 1483.0, 14.0, 2966.0])
        self.euclidean = np.array([0.21089672206, 0.00148105508243, 1.0, 0.00148583482109])
        self.lin = np.array([1.0, 0.995623769303, 1.0, 0.991242564101])
        
                #create the spaces required in the tests
        bcs.main(["build_core_space.py", 
          "-l", self.dir_ + "pipelines_test_resources/log1.txt",
          "-i", self.dir_ + "pipelines_test_resources/mat3",
          "-w", "raw",
          "-s", "top_sum_3",
          "-r", "svd_2", 
          "-o", self.dir_  + "pipelines_test_resources/",
          "--input_format", "dm"
          ])

@pytest.fixture
def space(tmpdir, data_dir, pipelines_test_resources):
    """Create the spaces required in the tests."""
    path = tmpdir.mkdir('space')

    bcs.main([
        "build_core_space.py",
        "-l", str(path.join('space_creation.log')),
        "-i", str(pipelines_test_resources.join('mat3')),
        "-w", "raw",
        "-s", "top_sum_3",
        "-r", "svd_2",
        "-o", str(path),
        "--input_format", "dm"
    ])

    return path


@pytest.fixture
def pkl_file(space):
    return str(space.join('CORE_SS.mat3.raw.top_sum_3.svd_2.pkl'))


@pytest.fixture(params=(
    (
        lambda sim_input, similarity, pkl_file, space, config_file, pipelines_test_resources: [
            "-i", sim_input,
            "-m", similarity,
            "-s", pkl_file,
            "-c", "1,2",
            "-o", str(space),
        ],
        'SIMS.sim_input.txt.CORE_SS.mat3.raw.top_sum_3.svd_2.{0}',
    ),
    (
        lambda sim_input, similarity, pkl_file, space, config_file, pipelines_test_resources: [
            "-i", sim_input,
            "--sim_measure", similarity,
            "--space", pkl_file,
            "--columns", "1,2",
            "-o", str(space),
        ],
        'SIMS.sim_input.txt.CORE_SS.mat3.raw.top_sum_3.svd_2.{0}',
    ),
    (
        lambda sim_input, similarity, pkl_file, space, config_file, pipelines_test_resources: [
            "--sim_measure", similarity,
            "--space", pkl_file,
            "--columns", "1,2",
            "-o", str(space),
            str(config_file),
        ],
        'SIMS.sim_input.txt.CORE_SS.mat3.raw.top_sum_3.svd_2.{0}',
    ),
    (
        lambda sim_input, similarity, pkl_file, space, config_file, pipelines_test_resources: [
            "--sim_measure", similarity,
            "--space", '{0},{0}'.format(pkl_file),
            "--columns", "1,2",
            "-o", str(space),
            str(config_file),
        ],
        'SIMS.sim_input.txt.CORE_SS.mat3.raw.top_sum_3.svd_2.CORE_SS.mat3.raw.top_sum_3.svd_2.{0}',
    ),
    (
        lambda sim_input, similarity, pkl_file, space, config_file, pipelines_test_resources: [
            "--sim_measure", similarity,
            "--in_dir", str(space),
            "--columns", "1,2",
            "-o", str(space),
            str(config_file),
            ],
        'SIMS.sim_input.txt.CORE_SS.mat3.raw.top_sum_3.svd_2.{0}'
    ),
))
def arguments_similarity_file(request):
    return request.param


@pytest.fixture
def config_file(config_dir, sim_input, similarity, space, pkl_file):
    config = (
        '[compute_similarities]\n'
        'input={input_}\n'
        'sim_measure={sim_measure}\n'
        'output={output}\n'
        'space={space},{space}\n'
        'columns=0,1\n'
        'log={log}\n'
    ).format(
        input_=sim_input,
        sim_measure=similarity,
        output=str(space),
        space=pkl_file,
        log=str(space.join('compute_similarities_log.txt')),
    )

    config_file = config_dir.join('sim_config.cfg')
    config_file.write(config)

    return config_file


@pytest.mark.parametrize(
    ('similarity', 'gold_array'),
    (
        ('cos', np.array([1.0, 0.585984772383, 1.0, 0.585984772383])),
        ('dot_prod', np.array([28.0, 1483.0, 14.0, 2966.0])),
        ('euclidean', np.array([0.21089672206, 0.00148105508243, 1.0, 0.00148583482109])),
        ('lin', np.array([1.0, 0.995623769303, 1.0, 0.991242564101])),
    ),
)
def test_compute_sim(sim_input, similarity, pkl_file, gold_array, arguments_similarity_file, space, config_file, pipelines_test_resources):
    arguments, similarity_file = arguments_similarity_file
    args = arguments(sim_input, similarity, pkl_file, space, config_file, pipelines_test_resources)
    sim_pipeline.main(['compute_similarities.py'] + args)

    path = str(space.join(similarity_file.format(similarity)))
    result_array = np.array(read_number_list(path, 3))

    np.testing.assert_array_almost_equal(result_array, gold_array, 5)

    return path


@pytest.fixture
def pkl_file(space):
    return str(space.join('CORE_SS.mat3.raw.top_sum_3.svd_2.pkl'))


@pytest.fixture(params=(
    (
        lambda sim_input, similarity, pkl_file, space, config_file, pipelines_test_resources: [
            "-i", sim_input,
            "-m", similarity,
            "-s", pkl_file,
            "-c", "1,2",
            "-o", str(space),
        ],
        'SIMS.sim_input.txt.CORE_SS.mat3.raw.top_sum_3.svd_2.{0}',
    ),
    (
        lambda sim_input, similarity, pkl_file, space, config_file, pipelines_test_resources: [
            "-i", sim_input,
            "--sim_measure", similarity,
            "--space", pkl_file,
            "--columns", "1,2",
            "-o", str(space),
        ],
        'SIMS.sim_input.txt.CORE_SS.mat3.raw.top_sum_3.svd_2.{0}',
    ),
    (
        lambda sim_input, similarity, pkl_file, space, config_file, pipelines_test_resources: [
            "--sim_measure", similarity,
            "--space", pkl_file,
            "--columns", "1,2",
            "-o", str(space),
            str(config_file),
        ],
        'SIMS.sim_input.txt.CORE_SS.mat3.raw.top_sum_3.svd_2.{0}',
    ),
    (
        lambda sim_input, similarity, pkl_file, space, config_file, pipelines_test_resources: [
            "--sim_measure", similarity,
            "--space", '{0},{0}'.format(pkl_file),
            "--columns", "1,2",
            "-o", str(space),
            str(config_file),
        ],
        'SIMS.sim_input.txt.CORE_SS.mat3.raw.top_sum_3.svd_2.CORE_SS.mat3.raw.top_sum_3.svd_2.{0}',
    ),
    (
        lambda sim_input, similarity, pkl_file, space, config_file, pipelines_test_resources: [
            "--sim_measure", similarity,
            "--in_dir", str(space),
            "--columns", "1,2",
            "-o", str(space),
            str(config_file),
            ],
        'SIMS.sim_input.txt.CORE_SS.mat3.raw.top_sum_3.svd_2.{0}'
    ),
))
def arguments_similarity_file(request):
    return request.param


@pytest.fixture
def config_file(config_dir, sim_input, similarity, space, pkl_file):
    config = (
        '[compute_similarities]\n'
        'input={input_}\n'
        'sim_measure={sim_measure}\n'
        'output={output}\n'
        'space={space},{space}\n'
        'columns=0,1\n'
        'log={log}\n'
    ).format(
        input_=sim_input,
        sim_measure=similarity,
        output=str(space),
        space=pkl_file,
        log=str(space.join('compute_similarities_log.txt')),
    )

    config_file = config_dir.join('sim_config.cfg')
    config_file.write(config)

    return config_file


@pytest.mark.parametrize(
    ('similarity', 'gold_array'),
    (
        ('cos', np.array([1.0, 0.585984772383, 1.0, 0.585984772383])),
        ('dot_prod', np.array([28.0, 1483.0, 14.0, 2966.0])),
        ('euclidean', np.array([0.21089672206, 0.00148105508243, 1.0, 0.00148583482109])),
        ('lin', np.array([1.0, 0.995623769303, 1.0, 0.991242564101])),
    ),
)
def test_compute_sim(sim_input, similarity, pkl_file, gold_array, arguments_similarity_file, space, config_file, pipelines_test_resources):
    arguments, similarity_file = arguments_similarity_file
    args = arguments(sim_input, similarity, pkl_file, space, config_file, pipelines_test_resources)
    sim_pipeline.main(['compute_similarities.py'] + args)

    path = str(space.join(similarity_file.format(similarity)))
    result_array = np.array(read_number_list(path, 3))

    np.testing.assert_array_almost_equal(result_array, gold_array, 5)
