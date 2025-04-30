import os
import pandas as pd
import plotly.express as px
import dash_bootstrap_components as dbc
from dash import dcc, html


def get_country_code(country):
	country_codes = {"France": "FRA", "Germany": "DEU", "Spain": "ESP"}
	return country_codes.get(country, "")


def get_geo_churn_data(dataframe):
	geo_data = []

	for country in ["France", "Germany", "Spain"]:
		country_df = dataframe[dataframe[f"Geography_{country}"] == 1]

		if len(country_df) > 0:
			total_customers = len(country_df)
			churned_customers = len(country_df[country_df["Exited"] == 1])
			churn_rate = churned_customers / total_customers

			avg_credit_score = country_df["CreditScore"].mean()
			avg_age = country_df["Age"].mean()
			avg_balance = country_df["Balance"].mean()
			avg_tenure = country_df["Tenure"].mean()
			active_member_rate = country_df["IsActiveMember"].mean()

			geo_data.append(
				{
					"Country": country,
					"iso_alpha": get_country_code(country),
					"Total_Customers": total_customers,
					"Churned_Customers": churned_customers,
					"Churn_Rate": churn_rate,
					"Avg_Credit_Score": avg_credit_score,
					"Avg_Age": avg_age,
					"Avg_Balance": avg_balance,
					"Avg_Tenure": avg_tenure,
					"Active_Member_Rate": active_member_rate,
				}
			)

	return pd.DataFrame(geo_data)


file_path = os.path.join(os.path.dirname(__file__), "../../data/Cleaned_Churn_modelling.csv")
df = pd.read_csv(file_path)

geo_churn_df = get_geo_churn_data(df)


def get_geo_churn_frame():
	return geo_churn_df


def get_home_page():
	return [
		html.Div(
			children=[
				html.H1("Overview"),
				html.P(
					"Observe patterns in your data at a glance with this overview page.",
					className="description",
					style={"margin-bottom": "4em"},
				),
				dbc.Container(
					[
						dbc.Row(
							[
								dbc.Col(
									[
										dbc.Card(
											[
												dbc.CardHeader(
													[
														html.P(
															"Explore customer churn patterns across different countries",
															className="card-subtitle text-muted",
														),
													]
												),
												dbc.CardBody(
													[
														dbc.Row(
															[
																dbc.Col(
																	[
																		html.Label(
																			"Select metric to visualise:",
																			className="font-weight-bold",
																		),
																		dcc.Dropdown(
																			id="metric-dropdown",
																			options=[
																				{
																					"label": "Churn Rate",
																					"value": "Churn_Rate",
																				},
																				{
																					"label": "Average Credit Score",
																					"value": "Avg_Credit_Score",
																				},
																				{
																					"label": "Average Age",
																					"value": "Avg_Age",
																				},
																				{
																					"label": "Average Balance",
																					"value": "Avg_Balance",
																				},
																				{
																					"label": "Average Tenure",
																					"value": "Avg_Tenure",
																				},
																				{
																					"label": "Active Member Rate",
																					"value": "Active_Member_Rate",
																				},
																			],
																			value="Churn_Rate",
																			clearable=False,
																			className="mb-4",
																		),
																	],
																	width=12,
																	lg=6,
																)
															]
														),
														dbc.Row(
															[
																dbc.Col(
																	[
																		html.Div(
																			[
																				dcc.Graph(
																					id="choropleth-map",
																					figure=create_choropleth_map(),
																					config={
																						"displayModeBar": True,
																						"scrollZoom": True,
																						"responsive": True,
																					},
																					className="map-container",
																				)
																			],
																			className="map-wrapper",
																		)
																	],
																	width=12,
																)
															]
														),
													]
												),
											],
											className="mb-4",
										)
									],
									width=12,
								)
							]
						),
						dbc.Row(
							[
								dbc.Col(
									[
										dbc.Card(
											[
												dbc.CardHeader("Country Details"),
												dbc.CardBody(
													html.Div(
														id="country-details",
														children=[
															html.P(
																"Click on a country in the map to see detailed metrics.",
																className="text-muted",
															)
														],
													)
												),
											]
										)
									],
									width=12,
									lg=6,
								),
								dbc.Col(
									[
										dbc.Card(
											[
												dbc.CardHeader("Churn Rate Comparison"),
												dbc.CardBody(
													dcc.Graph(
														id="churn-bar-chart",
														figure=px.bar(
															geo_churn_df,
															x="Country",
															y="Churn_Rate",
															text=geo_churn_df["Churn_Rate"].apply(lambda x: f"{x:.1%}"),
															color="Churn_Rate",
															color_continuous_scale="Reds",
															labels={"Churn_Rate": "Churn Rate"},
														).update_layout(
															yaxis_tickformat=".0%",
															title="Churn Rate by Country",
															yaxis_title="Churn Rate",
															xaxis_title="Country",
														),
													)
												),
											]
										)
									],
									width=12,
									lg=6,
								),
							],
							className="mt-4",
						),
					],
					fluid=True,
				),
			]
		)
	]


def create_choropleth_map(colour_metric="Churn_Rate"):
	if colour_metric == "Churn_Rate":
		colour_scale = "Reds"
		hover_template = "<b>%{customdata[0]}</b><br>Churn Rate: %{z:.1%}<br>Total Customers: %{customdata[1]}<br>Churned: %{customdata[2]}"
	else:
		colour_scale = "Blues"
		hover_template = f"<b>%{{customdata[0]}}</b><br>{colour_metric.replace('_', ' ')}: %{{z:.1f}}<br>Churn Rate: %{{customdata[3]:.1%}}"

	df = geo_churn_df
	fig = px.choropleth(
		df,
		locations="iso_alpha",
		color=colour_metric,
		custom_data=[
			df["Country"],
			df["Total_Customers"],
			df["Churned_Customers"],
			df["Churn_Rate"],
		],
		scope="europe",
		color_continuous_scale=colour_scale,
		labels={colour_metric: colour_metric.replace("_", " ")},
	)

	fig.update_traces(hovertemplate=hover_template)
	fig.update_layout(
		title=f"Customer Churn Analysis by Country - {colour_metric.replace('_', ' ')}",
		geo=dict(
			showcoastlines=True,
			projection_type="natural earth",
			center=dict(lat=47, lon=10),
			projection_scale=3.5,
			showocean=True,
			oceancolor="lightblue",
			showland=True,
			landcolor="#EAEAAE",
			showcountries=True,
			countrycolor="gray",
			bgcolor="rgba(0,0,0,0)",
		),
		height=600,
		width=1200,
		margin=dict(l=10, r=10, t=50, b=10),
		paper_bgcolor="rgba(0,0,0,0)",
		autosize=True,
	)

	if colour_metric == "Churn_Rate":
		fig.update_coloraxes(colorbar_tickformat=".0%")

	return fig
