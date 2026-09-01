complete -c winhash -f
complete -c winhash -s h -l help -d "Show usage"
complete -c winhash -n "__fish_use_subcommand" -a "run devices status stop help" -d "Subcommands"

complete -c winhash -n "__fish_seen_subcommand_from run" -s m -l mode -d "Hashcat mode" -a "0 1000 1001 1002 22000" 
complete -c winhash -n "__fish_seen_subcommand_from run" -l backend-device -d "Preferred backend device id" -a "1 2 3"
complete -c winhash -n "__fish_seen_subcommand_from run" -l status-timer -d "Hashcat status poll interval (seconds)" -a "5 10 15 20 30"
complete -c winhash -n "__fish_seen_subcommand_from run" -l force -d "Run even if camera stream is active"
complete -c winhash -n "__fish_seen_subcommand_from run" -F

complete -c winhash -n "__fish_seen_subcommand_from devices" -s a -l all -d "Alias for future options"
complete -c winhash -n "__fish_seen_subcommand_from devices" -f

complete -c winhash -n "__fish_seen_subcommand_from run; or __fish_seen_subcommand_from status; or __fish_seen_subcommand_from stop; or __fish_seen_subcommand_from devices; or __fish_seen_subcommand_from help" -l user -d "SSH user"
complete -c winhash -n "__fish_seen_subcommand_from run; or __fish_seen_subcommand_from status; or __fish_seen_subcommand_from stop; or __fish_seen_subcommand_from devices; or __fish_seen_subcommand_from help" -l host -d "SSH host"
complete -c winhash -n "__fish_seen_subcommand_from run; or __fish_seen_subcommand_from status; or __fish_seen_subcommand_from stop; or __fish_seen_subcommand_from devices; or __fish_seen_subcommand_from help" -l key -d "SSH private key file"
complete -c winhash -n "__fish_seen_subcommand_from run; or __fish_seen_subcommand_from status; or __fish_seen_subcommand_from stop; or __fish_seen_subcommand_from devices; or __fish_seen_subcommand_from help" -l hashcat -d "Remote hashcat executable path"
complete -c winhash -n "__fish_seen_subcommand_from run; or __fish_seen_subcommand_from status; or __fish_seen_subcommand_from stop; or __fish_seen_subcommand_from devices; or __fish_seen_subcommand_from help" -l workdir -d "Remote hashcat work directory"
