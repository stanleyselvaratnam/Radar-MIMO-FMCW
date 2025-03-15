# Convert a stream of 32-bit words to sequences of complex numbers.
# yves.piguet@csem.ch

import math


class IQConverter:
    """
    Conversion between 4 bytes and complex IQ + direction
    """

    def __init__(self, byteorder="big", signed=True, direction_mask=0x10000000):
        self.byteorder = byteorder
        self.signed = signed
        self.direction_mask = direction_mask

    def to_bytes(self, z, up):
        re = math.floor(0x4000 * (z.real % 1.0))
        im = math.floor(0x4000 * (z.imag % 1.0))
        word = re << 14 | im | (0 if up else self.direction_mask)
        return word.to_bytes(length=4, byteorder=self.byteorder, signed=False)

    def from_bytes(self, b):
        word = int.from_bytes(b, byteorder=self.byteorder, signed=False)
        # extract as unsigned
        re = (word >> 14) & 0x3fff  # I
        im = word & 0x3fff  # Q
        if self.signed:
            # convert to signed
            if re & 0x2000:
                re -= 0x4000
            if im & 0x2000:
                im -= 0x4000
        else:
            # unsigned: subtract middle value
            re -= 0x2000
            im -= 0x2000
        # convert to complex in +/- 1
        return (re + 1j * im) / 0x2000, (word & self.direction_mask) == 0


class IQStreamConverter:
    """
    Conversion from byte stream to lists of complex IQ + direction
    """

    def __init__(self, max_length=10240, byteorder="big", signed=True, decimation=1):
        self.direction_mask = 0x10000000
        self.max_length = max_length
        self.byteorder = byteorder
        self.signed = signed
        self.decimation = decimation
        self.request_new_converter = False  # set to True to force process to update iq_converter

    def byte_slicer(self, input_gen):
        """
        Generator converting input byte stream to a stream of byte arrays of 4 bytes.
        """

        byte_buf = b""
        for b in input_gen:
            byte_buf += b
            n_words = len(byte_buf) // 4
            for i in range(n_words):
                yield byte_buf[4 * i : 4 * i + 4]
            byte_buf = byte_buf[4 * n_words : ]

    def process(self, input_gen):
        """
        Read chunks of byte arrays from input_gen, convert them to sequences
        of 32-bit words, then to complex numbers, and yields sequences of up
        to max_length complex numbers.
        """

        iq_converter = IQConverter(byteorder=self.byteorder, signed=self.signed, direction_mask=self.direction_mask)
        direction = False
        vec = []
        count = 0

        for word in self.byte_slicer(input_gen):
            if self.request_new_converter:
                iq_converter = IQConverter(byteorder=self.byteorder, signed=self.signed, direction_mask=self.direction_mask)
                self.request_new_converter = False
            z, d = iq_converter.from_bytes(word)
            if  d == direction and len(vec) < self.max_length:
                vec.append(z)
            else:
                count += 1
                if count >= self.decimation:
                    yield vec, direction
                    count = 0
                vec = [z]
                direction = d
