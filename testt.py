import neptune

# same token as above
# make sure to put correct username in project_qualified_name
neptune.init('muralidharpettela/sandbox', api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiMmRjMDMwNzgtYmIzMi00NGIyLTk1M2YtOTYzMzI4YjA4NGI3In0=')

# create experiment in the project defined above
neptune.create_experiment()

# send some metrics to your experiment
neptune.send_metric('acc', 0.95)

neptune.stop()