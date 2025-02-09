import pytest

def test_valid_initialization():
    layer = _ConvNd(
        in_channels=4,
        out_channels=8,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding="same",
        dilation=(1, 1),
        transposed=False,
        output_padding=(0, 0),
        groups=1,
        bias=True,
        padding_mode="zeros",
    )
    assert layer.in_channels == 4
    assert layer.out_channels == 8
    assert layer.padding == "same"

def test_invalid_groups():
    with pytest.raises(ValueError, match="groups must be a positive integer"):
        _ConvNd(
            in_channels=4,
            out_channels=8,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding="same",
            dilation=(1, 1),
            transposed=False,
            output_padding=(0, 0),
            groups=0,
            bias=True,
            padding_mode="zeros",
        )

def test_in_channels_not_divisible_by_groups():
    with pytest.raises(ValueError, match="in_channels must be divisible by groups"):
        _ConvNd(
            in_channels=5,
            out_channels=8,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding="same",
            dilation=(1, 1),
            transposed=False,
            output_padding=(0, 0),
            groups=2,
            bias=True,
            padding_mode="zeros",
        )

def test_invalid_padding_string():
    with pytest.raises(ValueError, match="Invalid padding string"):
        _ConvNd(
            in_channels=4,
            out_channels=8,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding="invalid",
            dilation=(1, 1),
            transposed=False,
            output_padding=(0, 0),
            groups=1,
            bias=True,
            padding_mode="zeros",
        )

def test_invalid_padding_mode():
    with pytest.raises(ValueError, match="padding_mode must be one of"):
        _ConvNd(
            in_channels=4,
            out_channels=8,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding="same",
            dilation=(1, 1),
            transposed=False,
            output_padding=(0, 0),
            groups=1,
            bias=True,
            padding_mode="invalid_mode",
        )

def test_padding_same_with_stride_not_one():
    with pytest.raises(ValueError, match="padding='same' is not supported for strided convolutions"):
        _ConvNd(
            in_channels=4,
            out_channels=8,
            kernel_size=(3, 3),
            stride=(2, 1),
            padding="same",
            dilation=(1, 1),
            transposed=False,
            output_padding=(0, 0),
            groups=1,
            bias=True,
            padding_mode="zeros",
        )

class TestSoftmax(unittest.TestCase):
    def test_softmax_simple_vector(self):
        input_vector = [1, 2, 3]
        expected_output = np.exp(input_vector) / np.sum(np.exp(input_vector))
        result = softmax(input_vector)
        self.assertTrue(np.allclose(result, expected_output))
    
    def test_softmax_negative_values(self):
        input_vector = [-1, 0, 1]
        expected_output = np.exp(input_vector) / np.sum(np.exp(input_vector))
        result = softmax(input_vector)
        self.assertTrue(np.allclose(result, expected_output))
    
    def test_softmax_large_values(self):
        input_vector = [1000, 1001, 1002]
        expected_output = np.exp(input_vector - np.max(input_vector)) / np.sum(np.exp(input_vector - np.max(input_vector)))
        result = softmax(input_vector)
        self.assertTrue(np.allclose(result, expected_output))

    def test_softmax_2d_matrix(self):
        input_matrix = [[1, 2, 3], [1, 2, 3]]
        expected_output = np.exp(input_matrix) / np.sum(np.exp(input_matrix), axis=1, keepdims=True)
        result = softmax(input_matrix)
        self.assertTrue(np.allclose(result, expected_output))

    def test_softmax_invalid_input(self):
        input_tensor = np.random.rand(2, 2, 2) 
        with self.assertRaises(ValueError):
            softmax(input_tensor)