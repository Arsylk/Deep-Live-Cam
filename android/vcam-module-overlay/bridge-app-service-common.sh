#!/system/bin/sh

# Shared bounded startup for the unified companion APK. Credential-encrypted
# Termux files can become visible a few seconds before Android user 0 and the
# package manager accept foreground-service starts during boot.
APP_SERVICE_START_ATTEMPTS=30
APP_SERVICE_START_RETRY_SECONDS=2

start_app_foreground_service() {
    service_label=$1
    shift
    service_attempt=1
    while [ "$service_attempt" -le "$APP_SERVICE_START_ATTEMPTS" ]; do
        if am start-foreground-service --user 0 "$@" >/dev/null 2>&1; then
            echo "$service_label started after $service_attempt attempt(s)"
            return 0
        fi
        if [ "$service_attempt" -eq 1 ] || \
           [ $((service_attempt % 5)) -eq 0 ]; then
            echo "waiting for user 0/package service: $service_label attempt $service_attempt/$APP_SERVICE_START_ATTEMPTS"
        fi
        service_attempt=$((service_attempt + 1))
        sleep "$APP_SERVICE_START_RETRY_SECONDS"
    done
    echo "WARNING: $service_label is not available after $APP_SERVICE_START_ATTEMPTS attempts"
    return 1
}
