import sys
import ratemeter
import tcpclient
import math
import iqstreamconverter
import getopt
import time


def help(argv0, **kwargs):
    print(f"""
Usage: python3 {argv0} [options]

Options:
  --bytes     undecoded bytes
  --samples   decoded as complex numbers and sweep direction
  --seq       decoded as sweep sequences of complex numbers
  --server S  server IP address
  --u32       decoded as unsigned 32-bit words
""", **kwargs)

def byte_slicer(input_gen):

    byte_buf = b""
    for b in input_gen:
        byte_buf += b
        n_words = len(byte_buf) // 4
        for i in range(n_words):
            yield byte_buf[4 * i : 4 * i + 4]
        byte_buf = byte_buf[4 * n_words : ]

def toStringSIPrefix(x, format):
    try:
        prefix = math.floor(math.log10(x) / 3)
        if -4 <= prefix <= 4:
            prefix_str = ["p", "n", "µ", "m", "", "k", "M", "G", "T"][prefix + 4]
            x /= 10 ** (3 * prefix)
        else:
            prefix_str = ""
        return (format % (x, )) + prefix_str
    except ValueError:
        return format % (x, )

def main(argv):

    what = "bytes"

    try:
        options, args = getopt.getopt(argv[1:], "", [
            "bytes",
            "help",
            "samples",
            "seq",
            "server=",
            "u32",
        ])
        for name, value in options:
            if name == "--bytes":
                what = "bytes"
            elif name == "--help":
                help(argv[0])
                sys.exit(0)
            elif name == "--samples":
                what = "samples"
            elif name == "--seq":
                what = "seq"
            elif name == "--server":
                server = value
            elif name == "--u32":
                what = "u32"
    except Exception:
        help(file=sys.stderr)
        sys.exit(1)

    g = tcpclient.tcp_data_gen(server_address=server)
    r = ratemeter.RateMeter()

    t0 = time.time()
    if what == "bytes":
        for b in g:
            r.notify([len(b)])
            if time.time() > t0 + 0.1:
                print(f"\rrate: {toStringSIPrefix(r.get_rate()[0], '%.3f ')}B/s  ", end="")
                t0 = time.time()
    elif what == "u32":
        for u32 in byte_slicer(g):
            r.notify([1])
            if time.time() > t0 + 0.1:
                print(f"\rrate: {toStringSIPrefix(r.get_rate()[0], '%.3f ')}words/s  ", end="")
                t0 = time.time()
    elif what == "samples":
        iq_converter = iqstreamconverter.IQConverter(byteorder="little", direction_mask=0x10000000)
        t0 = time.time()
        for u32 in byte_slicer(g):
            z, d = iq_converter.from_bytes(u32)
            r.notify([1])
            if time.time() > t0 + 0.1:
                print(f"\rrate: {toStringSIPrefix(r.get_rate()[0], '%.3f ')}samples/s  ", end="")
                t0 = time.time()
    elif what == "seq":
        t0 = time.time()
        seqlen = 0
        c = iqstreamconverter.IQStreamConverter(byteorder="little")
        for seq, dir in c.process(g):
            r.notify([1])
            seqlen = 0.95 * seqlen + 0.05 * len(seq)
            if time.time() > t0 + 0.1:
                print(f"\rrate: {toStringSIPrefix(r.get_rate()[0], '%.3f ')}sequences/s  len={len(seq)}  ", end="")
                t0 = time.time()


if __name__ == "__main__":
    main(sys.argv)
