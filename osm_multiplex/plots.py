import pickle
import plotly.graph_objects as go


dist = pickle.load(open('./osm_multiplex/data/lstm_dist.pickle', 'rb'))
collected = pickle.load(open('./osm_multiplex/data/lstm_collected_target.pickle', 'rb'))
predicted = pickle.load(open('./osm_multiplex/data/lstm_predicted_target.pickle', 'rb'))

fig = go.Figure()

button_list = []
dist_length = len(dist)
counter = 0

for location, results in dist.items():
    threshold = results['threshold']
    del results['threshold']
    x_week=[str(week) for week, _ in results.items()]

    fig.add_trace(
        go.Scatter(x=x_week,
                   y=[result[1] for _, result in results.items()],
                   name=location,
                   mode='lines',
                   visible=False)
    )
    fig.add_trace(
        go.Scatter(x=x_week,
                   y=[threshold] * len(x_week),
                   name=location + ' threshold',
                   visible=False)
    )
    
    visible_list = [False] * dist_length * 2
    visible_list[counter] = True
    visible_list[counter + 1] = True
    counter += 2

    button_list.append(
        dict(
            label=location,
            method="update",
            args=[{"visible": visible_list},
                  {"title": location}]
        )
    )

fig.update_layout(
    updatemenus=[
        go.layout.Updatemenu(
            active=0,
            buttons=list(button_list)
        )
    ],
    xaxis=dict(type='category')
)

fig.show()
# def plot_dist_threshold():
#     dist = pickle.load(open('./osm_multiplex/data/lstm_dist.pickle', 'rb'))

#     plot_number = len(dist)
#     fig, ax = plt.subplots(plot_number)

#     plot_count = 0
#     for location, results in dist.items():
#         threshold = results['threshold']
#         del results['threshold']
#         x_week = [week for week, _ in results.items()]
#         y_dist = [result[1] for _, result in results.items()]
#         ax[plot_count].set_title(location)
#         ax[plot_count].plot(x_week, y_dist)
#         ax[plot_count].tick_params(labelrotation=90)
#         ax[plot_count].axhline(y=threshold)
#         plot_count += 1
#     plt.tight_layout()