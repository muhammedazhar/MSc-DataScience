import os
import logging
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict
from IPython.display import display
import requests
import networkx as nx
import matplotlib.pyplot as plt
import folium
from geopy.distance import geodesic
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class LondonTubeNetwork:
    """
    Comprehensive implementation of London Underground network visualization.
    Includes data fetching, filtering, and multiple visualization options.
    """

    def __init__(self, app_key: str):
        """Initialize with TfL API key"""
        self.base_url = 'https://api.tfl.gov.uk'
        self.app_key = app_key
        self.G = nx.Graph()
        # Official TfL line colors
        self.line_colors = {
            'bakerloo': '#B36305',
            'central': '#E32017',
            'circle': '#FFD300',
            'district': '#00782A',
            'hammersmith-city': '#F3A9BB',
            'jubilee': '#A0A5A9',
            'metropolitan': '#9B0056',
            'northern': '#000000',
            'piccadilly': '#003688',
            'victoria': '#0098D4',
            'waterloo-city': '#95CDBA'
        }
        # Add station colors
        self.station_colors = {
            'regular': 'yellow',
            'interchange': 'red',
            'edge': 'black'
        }

    def fetch_tube_data(self) -> Tuple[Optional[dict], Optional[dict]]:
        """Fetch station and line data from TfL API"""
        try:
            # Fetch stations
            response = requests.get(
                f'{self.base_url}/StopPoint/Mode/tube',
                params={'app_key': self.app_key}
            )
            response.raise_for_status()
            stations = response.json()

            # Fetch lines
            response = requests.get(
                f'{self.base_url}/Line/Mode/tube',
                params={'app_key': self.app_key}
            )
            response.raise_for_status()
            lines = response.json()

            return stations, lines

        except requests.RequestException as e:
            logger.error(f"Error fetching tube data: {e}")
            return None, None

    def analyze_network_data(self, stations: dict, lines: dict) -> Optional[dict]:
        """Analyze and filter network data to meet coursework requirements"""
        station_lines = defaultdict(set)
        line_stations = defaultdict(set)

        # Process stations
        valid_stations = {}
        for station in stations['stopPoints']:
            if station['stopType'] == 'NaptanMetroStation' and 'tube' in station['modes']:
                station_id = station['naptanId']
                valid_stations[station_id] = {
                    'name': station['commonName'],
                    'lat': station['lat'],
                    'lon': station['lon']
                }

        # Process lines and their stations
        for line in lines:
            line_id = line['id']
            try:
                response = requests.get(
                    f'{self.base_url}/Line/{line_id}/Route/Sequence/all',
                    params={'app_key': self.app_key}
                )
                response.raise_for_status()
                line_data = response.json()

                if 'orderedLineRoutes' in line_data:
                    for route in line_data['orderedLineRoutes']:
                        for station_id in route['naptanIds']:
                            if station_id in valid_stations:
                                station_lines[station_id].add(line_id)
                                line_stations[line_id].add(station_id)

            except requests.RequestException as e:
                logger.warning(f"Error fetching data for line {line_id}: {e}")

        # Filter lines with sufficient stations
        valid_lines = {
            line_id: stations
            for line_id, stations in line_stations.items()
            if len(stations) >= 5
        }

        if len(valid_lines) >= 5:
            # Select top 5 lines with most stations
            selected_lines = dict(
                sorted(valid_lines.items(),
                       key=lambda x: len(x[1]),
                       reverse=True)[:5]
            )

            # Get stations for selected lines
            selected_stations = set()
            for stations in selected_lines.values():
                selected_stations.update(stations)

            return {
                'stations': {
                    station_id: {
                        **valid_stations[station_id],
                        'lines': station_lines[station_id]
                    }
                    for station_id in selected_stations
                },
                'lines': selected_lines,
                'stats': {
                    'num_stations': len(selected_stations),
                    'num_lines': len(selected_lines),
                    'lines_info': {
                        line: len(stations)
                        for line, stations in selected_lines.items()
                    }
                }
            }

        return None

    def build_filtered_network(self) -> bool:
        """Build network with filtered data meeting requirements"""
        stations, lines = self.fetch_tube_data()
        if not stations or not lines:
            return False

        analysis = self.analyze_network_data(stations, lines)
        if not analysis:
            logger.error("Insufficient data meeting requirements")
            return False

        # Log analysis results
        logger.info("\nNetwork Analysis:")
        logger.info("Selected Lines:")
        for line, stations in analysis['lines'].items():
            logger.info(f"- {line}: {len(stations)} stations")
        logger.info(f"Total Stations: {analysis['stats']['num_stations']}")

        # Build network with filtered data
        self._build_network_from_analysis(analysis)
        return True

    def _build_network_from_analysis(self, analysis: dict) -> None:
        """Build network from analyzed data"""
        # Add stations
        for station_id, station_data in analysis['stations'].items():
            self.G.add_node(
                station_id,
                name=station_data['name'],
                pos=(station_data['lon'], station_data['lat']),
                lines=station_data['lines']
            )

        # Add connections
        for line_id in analysis['lines'].keys():
            try:
                response = requests.get(
                    f'{self.base_url}/Line/{line_id}/Route/Sequence/all',
                    params={'app_key': self.app_key}
                )
                response.raise_for_status()
                line_data = response.json()

                if 'orderedLineRoutes' in line_data:
                    for route in line_data['orderedLineRoutes']:
                        stations = route['naptanIds']
                        for i in range(len(stations) - 1):
                            self._add_connection(
                                stations[i],
                                stations[i + 1],
                                line_id
                            )

            except requests.RequestException as e:
                logger.error(f"Error fetching line data for {line_id}: {e}")

    def _add_connection(self, start_station: str, end_station: str, line_id: str) -> None:
        """Add a connection between stations"""
        if (self.G.has_node(start_station)
            and self.G.has_node(end_station)
                and not self.G.has_edge(start_station, end_station)):

            # Calculate distance
            start_pos = self.G.nodes[start_station]['pos']
            end_pos = self.G.nodes[end_station]['pos']
            distance = geodesic(
                (start_pos[1], start_pos[0]),
                (end_pos[1], end_pos[0])
            ).kilometers

            # Add edge
            self.G.add_edge(
                start_station,
                end_station,
                line=line_id,
                color=self.line_colors.get(line_id.lower(), '#808080'),
                distance=round(distance, 2)
            )

    def create_schematic_map(self) -> plt.Figure:
        """Create schematic network visualization"""
        plt.figure(figsize=(15, 10))

        # Use geographic coordinates
        pos = {node: data['pos'] for node, data in self.G.nodes(data=True)}

        # Draw edges (lines)
        drawn_lines = set()
        for (u, v, data) in self.G.edges(data=True):
            plt.plot(
                [pos[u][0], pos[v][0]],
                [pos[u][1], pos[v][1]],
                color=data['color'],
                linewidth=2,
                alpha=0.7,
                label=data['line'] if data['line'] not in drawn_lines else "",
                zorder=1
            )
            drawn_lines.add(data['line'])

        # Draw stations
        for node, data in self.G.nodes(data=True):
            is_interchange = len(data['lines']) > 1
            plt.plot(
                data['pos'][0],
                data['pos'][1],
                'o',
                color=self.station_colors['interchange'] if is_interchange else self.station_colors['regular'],
                markeredgecolor=self.station_colors['edge'],
                markersize=12 if is_interchange else 8,
                markeredgewidth=2 if is_interchange else 1,
                zorder=2
            )

            # Add station labels
            plt.annotate(
                data['name'],
                (data['pos'][0], data['pos'][1]),
                xytext=(10, 10),
                textcoords='offset points',
                fontsize=8,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7),
                zorder=3
            )

        plt.title("London Underground Network (Selected Lines)",
                  pad=20, fontsize=16)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.axis('off')
        return plt

    def create_folium_map(self) -> folium.Map:
        """Create interactive network visualization"""
        m = folium.Map(
            location=[51.5074, -0.1278],
            zoom_start=12,
            tiles='cartodbpositron'
        )

        # Add lines
        for (u, v, data) in self.G.edges(data=True):
            line_coords = [
                [self.G.nodes[u]['pos'][1], self.G.nodes[u]['pos'][0]],
                [self.G.nodes[v]['pos'][1], self.G.nodes[v]['pos'][0]]
            ]

            # Create popup
            popup_html = f"""
                <div style="font-family: Arial, sans-serif;">
                    <h4>{data['line']} Line</h4>
                    <p>Distance: {data['distance']:.2f}km</p>
                    <p>Stations: {self.G.nodes[u]['name']} → {self.G.nodes[v]['name']}</p>
                </div>
            """

            folium.PolyLine(
                line_coords,
                color=data['color'],
                weight=3,
                opacity=0.7,
                popup=folium.Popup(popup_html, max_width=300)
            ).add_to(m)

        # Add stations
        for node, data in self.G.nodes(data=True):
            is_interchange = len(data['lines']) > 1

            popup_html = f"""
                <div style="font-family: Arial, sans-serif;">
                    <h4>{data['name']}</h4>
                    <p>Lines: {', '.join(data['lines'])}</p>
                    <p>Type: {'Interchange' if is_interchange else 'Regular'} Station</p>
                </div>
            """

            folium.CircleMarker(
                location=[data['pos'][1], data['pos'][0]],
                radius=8 if is_interchange else 5,
                color=self.station_colors['edge'],
                fill=True,
                fill_color=self.station_colors['interchange'] if is_interchange else self.station_colors['regular'],
                weight=2 if is_interchange else 1,
                popup=folium.Popup(popup_html, max_width=300)
            ).add_to(m)

        return m

    def visualize(self, output_path: Optional[str] = None) -> None:
        """Create both static and interactive visualizations"""

        # Create and display interactive map
        m = self.create_folium_map()
        display(m)

        # Optionally save to file if path provided
        if output_path:
            m.save(output_path)

    def get_network_stats(self) -> Dict:
        """Get network statistics"""
        return {
            'num_stations': self.G.number_of_nodes(),
            'num_connections': self.G.number_of_edges(),
            'num_interchanges': sum(
                1 for _, data in self.G.nodes(data=True)
                if len(data['lines']) > 1
            ),
            'total_distance': sum(
                data['distance']
                for _, _, data in self.G.edges(data=True)
            ),
            'is_connected': nx.is_connected(self.G)
        }


def main():
    """Main execution function"""
    load_dotenv('../Docs/.env')
    app_key = os.getenv('app_key')

    if not app_key:
        logger.error("No TfL API key found")
        return

    # Create and build filtered network
    network = LondonTubeNetwork(app_key)
    if network.build_filtered_network():
        # Print statistics
        stats = network.get_network_stats()
        logger.info("\nNetwork Statistics:")
        for key, value in stats.items():
            logger.info(f"{key}: {value}")

        # Create visualizations
        network.visualize('TfL-Map.html')
    else:
        logger.error("Failed to build network")

if __name__ == "__main__":
    main()
