import base64
import io
import os

import dash_bootstrap_components as dbc
import joblib
import pandas as pd
from dash import Dash, Input, Output, State, dash_table, dcc, html
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import models

from pages.page_404 import get_404_page
from pages.page_home import create_choropleth_map, get_geo_churn_frame, get_home_page
from pages.page_predict import get_predict_page

user_df = pd.DataFrame()

app = Dash(external_stylesheets=[dbc.themes.LUMEN, dbc.icons.FONT_AWESOME], suppress_callback_exceptions=True)
server = app.server
app.layout = [
	html.Link(rel="stylesheet", href="styles.css"),
	dcc.Location(id="url", refresh=False),
	html.Div(
		children=[
			html.Div(
				children=[
					html.A(
						children=[
							html.I(
								className="fa-solid fa-meteor",
								style={"font-size": "3em", "color": "red", "margin-bottom": "1em"},
							)
						],
						href="/",
					),
					dbc.Nav(
						id="nav-items",
						pills=True,
						style={"gap": "1em", "display": "flex", "flex-direction": "column"},
					),
				],
				style={
					"display": "flex",
					"flex-direction": "column",
					"max-width": "10%",
					"align-items": "center",
					"gap": "2em",
					"height": "50vh",
					"position": "fixed",
				},
			),
			html.Div(id="page-content", children=[], style={"width": "100%", "margin-left": "10%"}),
		],
		style={"display": "flex", "flex-direciton": "row", "gap": "5em"},
		className="page-container",
	),
]


@app.callback(Output("choropleth-map", "figure"), [Input("metric-dropdown", "value")])
def update_map(selected_metric):
	return create_choropleth_map(selected_metric)


@app.callback(Output("country-details", "children"), [Input("choropleth-map", "clickData")])
def display_country_details(clickData):
	if not clickData:
		return html.P("Click on a country in the map to see detailed metrics.", className="text-muted")

	loc = clickData["points"][0]["location"]
	iso_to_country = {"FRA": "France", "DEU": "Germany", "ESP": "Spain"}
	country = next((c for iso, c in iso_to_country.items() if iso == loc), None)

	if not country:
		return html.P("Country data not available", className="text-danger")

	geo_churn_df = get_geo_churn_frame()
	country_data = geo_churn_df[geo_churn_df["Country"] == country].iloc[0]

	return [
		html.H4(country, className="card-title"),
		dbc.Table(
			[
				html.Thead(html.Tr([html.Th("Metric"), html.Th("Value")])),
				html.Tbody(
					[
						html.Tr([html.Td("Total Customers"), html.Td(f"{country_data['Total_Customers']:,}")]),
						html.Tr([html.Td("Churned Customers"), html.Td(f"{country_data['Churned_Customers']:,}")]),
						html.Tr([html.Td("Churn Rate"), html.Td(f"{country_data['Churn_Rate']:.1%}")]),
						html.Tr([html.Td("Avg. Credit Score"), html.Td(f"{country_data['Avg_Credit_Score']:.1f}")]),
						html.Tr([html.Td("Avg. Age"), html.Td(f"{country_data['Avg_Age']:.1f}")]),
						html.Tr([html.Td("Avg. Balance"), html.Td(f"${country_data['Avg_Balance']:,.2f}")]),
						html.Tr([html.Td("Avg. Tenure"), html.Td(f"{country_data['Avg_Tenure']:.1f} years")]),
						html.Tr([html.Td("Active Member Rate"), html.Td(f"{country_data['Active_Member_Rate']:.1%}")]),
					]
				),
			],
			bordered=True,
			hover=True,
			striped=True,
			size="sm",
		),
	]


def to_binary(value):
	return 1 if value == "1" else 0


@app.callback(
	[
		Output("predict-output", "children"),
		Output("eval-toast", "is_open"),
		Output("eval-toast", "header"),
		Output("eval-toast", "children"),
		Output("eval-toast", "icon"),
	],
	Input("data-table", "active_cell"),
	State("model-select", "value"),
	State("data-table", "data"),
)
def cell_select(active_cell, model_select, table_data):
	if not active_cell:
		return [None, False, None, None, None]

	if not model_select or not active_cell:
		return [
			None,
			True,
			"Model Selection",
			"Please select a model",
			"danger",
		]

	selected_row = active_cell["row"]
	row_data = table_data[selected_row]
	credit = row_data.get("CreditScore", None)
	gender = row_data.get("Gender", None)
	age = row_data.get("Age", None)
	tenure = row_data.get("Tenure", None)
	balance = row_data.get("Balance", None)
	num_prod = row_data.get("NumOfProducts", None)
	has_card = row_data.get("HasCrCard", None)
	active_member = row_data.get("IsActiveMember", None)
	salary = row_data.get("EstimatedSalary", None)
	country_france = row_data.get("Geography_France", None)
	country_germany = row_data.get("Geography_Germany", None)
	country_spain = row_data.get("Geography_Spain", None)

	if not all(
		[
			credit is not None,
			gender is not None,
			age is not None,
			tenure is not None,
			balance is not None,
			num_prod is not None,
			has_card is not None,
			active_member is not None,
			salary is not None,
			country_france is not None,
			country_germany is not None,
			country_spain is not None,
		]
	):
		return [
			html.P("Data is missing required fields.", className="text-danger"),
			True,
			"Missing Fields",
			"Selected row is missing required fields.",
			"danger",
		]

	input_df = pd.DataFrame(
		[
			[
				int(credit),
				to_binary(gender),
				int(age),
				int(tenure),
				float(balance),
				int(num_prod),
				to_binary(has_card),
				float(active_member),
				int(salary),
				to_binary(country_france == "FRA"),
				to_binary(country_germany == "DEU"),
				to_binary(country_spain == "ESP"),
			]
		],
		columns=[
			"CreditScore",
			"Gender",
			"Age",
			"Tenure",
			"Balance",
			"NumOfProducts",
			"HasCrCard",
			"IsActiveMember",
			"EstimatedSalary",
			"Geography_France",
			"Geography_Germany",
			"Geography_Spain",
		],
	)

	model_map = {
		"ann": {
			"name": "Artifical Neural Network",
			"model_path": "ann.keras",
		},
		"random_forest": {
			"name": "Random Forest",
			"model_path": "random_forest.pkl",
		},
	}

	selected_model = model_map.get(model_select)
	file_dir = os.path.dirname(__file__)
	model_name = selected_model.get("name")
	model_path = os.path.join(file_dir, "models", selected_model.get("model_path"))
	model = load_model(model_path)
	churn_probability = predict_customer_churn(model, model_path, input_df)

	return [
		dbc.Card(
			[
				dbc.CardHeader(html.P("Model Prediction")),
				dbc.CardBody(
					[
						html.Span(children="Churn Probability", style={"font-weight": "bold"}),
						html.P(f"{churn_probability} %", style={"font-weight": "bold", "font-size": "1.5em"}),
					]
				),
			]
		),
		True,
		model_name,
		f"Customer churn prediction successfully completed with {model_name} model.",
		"success",
	]


def predict_customer_churn(model, model_path, input_df):
	if model_path.endswith(".keras"):
		scaler = joblib.load(os.path.join(os.path.dirname(__file__), "models/scaler.pkl"))
		input_df = scaler.transform(input_df)
		prediction = model.predict(input_df)
		prediction_value = prediction[0][0]
	elif model_path.endswith(".pkl"):
		prediction = model.predict_proba(input_df)
		prediction_value = prediction[0][1]

	churn_probability = round(float(prediction_value) * 100, 2)
	return churn_probability


def load_model(model_path):
	if not os.path.exists(model_path):
		return None

	if model_path.endswith(".keras"):
		return models.load_model(model_path)

	if model_path.endswith(".pkl"):
		return joblib.load(model_path)


@app.callback(
	Output("predict-input", "children"),
	Input("model-select", "value"),
	prevent_initial_call=True,
)
def get_predict_input(value):
	return [
		dcc.Upload(
			id="upload-data",
			children=["Drag and Drop or ", html.A("Click to Upload A CSV File")],
			style={
				"user-select": "none",
				"padding": "0.5em",
				"font-weight": "600",
				"text-align": "center",
				"cursor": "pointer",
				"border": "2px dashed #ccc",
				"border-radius": "10px",
				"stroke": "none",
			},
		)
	]


@app.callback(
	Output("upload-table", "children"),
	Input("upload-data", "contents"),
	State("upload-data", "filename"),
	State("upload-data", "last_modified"),
)
def upload_file(contents, filename, date):
	if not contents:
		return None

	user_df = parse_upload(contents, filename, date)
	return [
		dbc.Card(
			[
				dbc.CardHeader(html.P("Uploaded Data")),
				dbc.CardBody(
					[
						dash_table.DataTable(
							user_df.to_dict("records"),
							[{"name": i, "id": i} for i in user_df.columns],
							page_size=20,
							id="data-table",
							style_table={"overflowX": "auto"},
						)
					]
				),
			]
		)
	]


def parse_upload(contents, filename, date):
	content_type, content_string = contents.split(",")
	if "csv" not in content_type or "csv" not in filename:
		return None

	decoded = base64.b64decode(content_string)
	return pd.read_csv(io.StringIO(decoded.decode("utf-8")))


@app.callback(Output("page-content", "children"), Input("url", "pathname"))
def render_content(pathname):
	if pathname == "/predict":
		return get_predict_page()
	else:
		return get_home_page() if pathname == "/" else get_404_page()


@app.callback(Output("nav-items", "children"), Input("url", "pathname"))
def get_nav_items(pathname):
	if pathname == "/predict":
		return [
			dbc.NavItem(
				dbc.NavLink(
					children=html.I(className="fa-solid fa-home", style={"font-size": "1.3em", "color": "red"}),
					href="/",
				),
			),
			dbc.NavItem(
				dbc.NavLink(
					children=html.I(
						className="fa-solid fa-rectangle-list", style={"font-size": "1.3em", "color": "red"}
					),
					href="/predict",
					active=True,
				),
			),
		]
	else:
		return [
			dbc.NavItem(
				dbc.NavLink(
					children=html.I(className="fa-solid fa-home", style={"font-size": "1.3em", "color": "red"}),
					href="/",
					active=True,
				),
			),
			dbc.NavItem(
				dbc.NavLink(
					children=html.I(
						className="fa-solid fa-rectangle-list", style={"font-size": "1.3em", "color": "red"}
					),
					href="/predict",
				),
			),
		]


if __name__ == "__main__":
	# https://render.com/docs/environment-variables#all-runtimes
	app.run(host="0.0.0.0", debug=os.environ.get("RENDER", "false") != "true")
