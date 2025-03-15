# tcpclient.py
# generator for TCP data stream

import socket

def tcp_data_gen(server_address="localhost", server_port=9999, chunk_max_len=1024, verbose=False):
    """
    Connect to a server, read raw data from it and yield it in byte chunks
    of unspecified length, as expected by IQStreamConverter.process
    """

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    try:
        s.connect((server_address, server_port))
        if verbose:
            print(f"connected to {server_address}:{server_port}")
        while True:
            data = s.recv(chunk_max_len)
            if verbose:
                print(f"data chunk, len={len(data)}, data[0]={data[0]}")
            yield data
    finally:
        s.close()
