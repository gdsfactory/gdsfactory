# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: base
#     language: python
#     name: python3
# ---

# %% [markdown]
# # gdsfactory downloads
#
# You can plot the recent downloads for gdsfactory.

# %%
import requests
import datetime
import plotly.graph_objects as go


def get_total_downloads(package_name):
    statistics = []
    end_date = datetime.date.today()

    while True:
        url = f"https://pypistats.org/api/packages/{package_name}/overall"
        response = requests.get(url, params={"last_day": end_date})
        data = response.json()

        if response.status_code == 200:
            statistics.extend(
                [(entry["date"], entry["downloads"]) for entry in data["data"]]
            )
            if "next_day" in data:
                end_date = data["next_day"]
            else:
                break
        else:
            return None

    statistics.sort(key=lambda x: x[0])  # Sort by date
    dates, downloads = zip(*statistics)
    cumulative_downloads = [sum(downloads[: i + 1]) for i in range(len(downloads))]

    return dates, cumulative_downloads


# Replace 'gdsfactory' with the package you want to check
package_name = "gdsfactory"
dates, cumulative_downloads = get_total_downloads(package_name)

if dates and cumulative_downloads:
    fig = go.Figure(data=go.Scatter(x=dates, y=cumulative_downloads))
    fig.update_layout(
        xaxis=dict(title="Date", tickformat="%Y-%m-%d", tickangle=45, showgrid=True),
        yaxis=dict(title="Total Downloads", showgrid=True),
        title=f"Total Downloads - {package_name}",
    )
    fig.show()
else:
    print(f"Failed to retrieve download statistics for package '{package_name}'.")


# %%
