import dash_bootstrap_components as dbc

from dash import html


def get_predict_page():
	return [
		html.Div(
			children=[
				dbc.Toast(
					"This toast is placed in the top right",
					id="eval-toast",
					header="Customer Churn Prediction",
					is_open=False,
					dismissable=True,
					duration=4000,
					icon="success",
					style={"position": "fixed", "top": 40, "right": 40, "width": 350, "z-index": "99"},
				),
				html.H1("Churn Prediction"),
				html.P(
					"Input new data with a select machine learning model to predict whether a customer will churn.",
					className="description",
				),
				html.Div(
					children=[
						html.Div(
							children=[
								html.Span(children="SELECT MODEL", style={"font-weight": "bold", "font-size": "0.6em"}),
								dbc.Select(
									id="model-select",
									options=[
										{"label": "Artificial Neural Network (Most Accurate)", "value": "ann"},
										{"label": "Random Forest", "value": "random_forest"},
									],
									style={"max-width": "20em"},
								),
							]
						),
						html.Div(
							children=[
								html.Div(
									id="predict-input",
									style={
										"max-width": "22em",
										"font-size": "0.8em",
										"margin-top": "1em",
									},
								),
							]
						),
					],
					style={
						"display": "flex",
						"flex-direction": "row",
						"margin-bottom": "2em",
						"gap": "4em",
						# "justify-content": "center",
						"align-items": "center",
					},
				),
				html.Div(
					children=[
						html.Div(id="upload-table", style={"flex": "1", "width": "50%"}),
						html.Div(
							children=[
								html.Div(id="predict-output"),
								# html.Div(id="predict-graph"),
							],
							style={
								"display": "flex",
								"flex-direction": "column",
								"gap": "2em",
								"flex": "1",
								"width": "50%",
							},
						),
					],
					style={
						"display": "flex",
						"justify-content": "space-between",
						"gap": "2em",
						"flex-direction": "row",
						"width": "100%",
					},
				),
			],
			style={"width": "100%"},
		)
	]
