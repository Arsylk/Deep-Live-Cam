function winhash --description "SSH-deploy hashcat jobs to Windows RTX host"
    if test (count $argv) -eq 0
        echo "usage: winhash <run|devices|status|stop|help> ..."
        echo "examples:"
        echo "  winhash run hashes.txt -m 0"
        echo "  winhash run --force hashfile.txt -a 0 -m 1000 -r rules\best64.rule hashes.txt"
        echo "  winhash devices"
        echo "  winhash status"
        echo "  winhash stop"
        return 1
    end

    set -l client /opt/github/Deep-Live-Cam/tools/winhash_client.py
    if not test -f $client
        echo "Missing client script at $client" >&2
        return 1
    end

    switch $argv[1]
        case h -h --help help
            python3 $client --help
            return $status
    end

    python3 $client $argv
end
