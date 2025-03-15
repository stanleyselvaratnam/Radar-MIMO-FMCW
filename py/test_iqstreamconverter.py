import iqstreamconverter

def data_gen(seq_len=10, chunk_len=500):
    """

    """
    byte_buf = b""
    seq_count = 0
    direction = True
    c = iqstreamconverter.IQConverter()
    while True:
        byte_buf += c.to_bytes(0, direction)
        seq_count += 1
        if seq_count >= seq_len:
            seq_count = 0
            direction = not direction
        if len(byte_buf) >= chunk_len:
            yield byte_buf[ : chunk_len]
            byte_buf = byte_buf[chunk_len : ]

c = iqstreamconverter.IQStreamConverter()

for seq, dir in c.process(data_gen()):
    print(seq, dir)
