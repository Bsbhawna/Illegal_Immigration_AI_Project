def filter_alerts(alerts, selected_sources, selected_severities):
    """
    Filter alerts based on selected sources and severity.
    """
    return [
        alert for alert in alerts
        if alert["source"] in selected_sources and alert["severity"] in selected_severities
    ]
