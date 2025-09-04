import React, { useState } from 'react';
import { BarChart, TrendingUp, Database, Calendar, Hash, Type, AlertCircle, Filter } from 'lucide-react';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Separator } from '@/components/ui/separator';
import { Checkbox } from '@/components/ui/checkbox';
import { Button } from '@/components/ui/button';
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover';
import type { InsightsData, ColumnInsight } from '@/lib/api';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';

// Status filter component for cross-column results
function StatusFilter({ 
  selectedStatuses, 
  onStatusChange 
}: { 
  selectedStatuses: Set<string>; 
  onStatusChange: (statuses: Set<string>) => void; 
}) {
  const statusOptions = [
    { value: 'high', label: 'High', color: 'text-red-600' },
    { value: 'medium', label: 'Medium', color: 'text-orange-600' },
    { value: 'low', label: 'Low', color: 'text-yellow-600' },
    { value: 'skipped', label: 'Skipped', color: 'text-gray-600' }
  ];

  const handleStatusToggle = (status: string, checked: boolean) => {
    const newStatuses = new Set(selectedStatuses);
    if (checked) {
      newStatuses.add(status);
    } else {
      newStatuses.delete(status);
    }
    onStatusChange(newStatuses);
  };

  const selectedCount = selectedStatuses.size;
  const totalCount = statusOptions.length;

  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button variant="outline" size="sm" className="gap-2">
          <Filter className="w-4 h-4" />
          Filter Status
          {selectedCount < totalCount && (
            <Badge variant="secondary" className="ml-1 px-1.5 py-0.5 text-xs">
              {selectedCount}
            </Badge>
          )}
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-56" align="end">
        <div className="space-y-3">
          <div className="font-medium text-sm">Show Status Levels</div>
          <div className="space-y-2">
            {statusOptions.map((option) => (
              <div key={option.value} className="flex items-center space-x-2">
                <Checkbox
                  id={option.value}
                  checked={selectedStatuses.has(option.value)}
                  onCheckedChange={(checked) => handleStatusToggle(option.value, !!checked)}
                />
                <label 
                  htmlFor={option.value} 
                  className={`text-sm font-medium cursor-pointer ${option.color}`}
                >
                  {option.label}
                </label>
              </div>
            ))}
          </div>
          <Separator />
          <div className="flex justify-between">
            <Button 
              variant="ghost" 
              size="sm" 
              onClick={() => onStatusChange(new Set(statusOptions.map(o => o.value)))}
            >
              Select All
            </Button>
            <Button 
              variant="ghost" 
              size="sm" 
              onClick={() => onStatusChange(new Set(['high']))}
            >
              Reset
            </Button>
          </div>
        </div>
      </PopoverContent>
    </Popover>
  );
}

// StatusBadge component for cross-column rule status indicators
function StatusBadge({ status }: { status: string }) {
  const getStatusStyle = (status: string) => {
    const normalizedStatus = status.toLowerCase();
    
    switch (normalizedStatus) {
      case 'high':
        return 'bg-red-200 text-red-900 border-red-300 hover:bg-red-250 dark:bg-red-800/60 dark:text-red-100 dark:border-red-600 dark:hover:bg-red-800/70';
      case 'medium':
        return 'bg-orange-200 text-orange-900 border-orange-300 hover:bg-orange-250 dark:bg-orange-800/60 dark:text-orange-100 dark:border-orange-600 dark:hover:bg-orange-800/70';
      case 'low':
        return 'bg-yellow-200 text-yellow-900 border-yellow-300 hover:bg-yellow-250 dark:bg-yellow-700/60 dark:text-yellow-100 dark:border-yellow-600 dark:hover:bg-yellow-700/70';
      case 'skipped':
        return 'bg-gray-200 text-gray-800 border-gray-300 hover:bg-gray-250 dark:bg-gray-700/60 dark:text-gray-200 dark:border-gray-600 dark:hover:bg-gray-700/70';
      // Backward compatibility for old status names
      case 'passed':
        return 'bg-yellow-200 text-yellow-900 border-yellow-300 hover:bg-yellow-250 dark:bg-yellow-700/60 dark:text-yellow-100 dark:border-yellow-600 dark:hover:bg-yellow-700/70';
      case 'warning':
        return 'bg-orange-200 text-orange-900 border-orange-300 hover:bg-orange-250 dark:bg-orange-800/60 dark:text-orange-100 dark:border-orange-600 dark:hover:bg-orange-800/70';
      case 'error':
      case 'failed':
        return 'bg-red-200 text-red-900 border-red-300 hover:bg-red-250 dark:bg-red-800/60 dark:text-red-100 dark:border-red-600 dark:hover:bg-red-800/70';
      default:
        return 'bg-gray-200 text-gray-800 border-gray-300 hover:bg-gray-250 dark:bg-gray-700/60 dark:text-gray-200 dark:border-gray-600 dark:hover:bg-gray-700/70';
    }
  };

  return (
    <span 
      className={`inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-semibold transition-colors capitalize ${getStatusStyle(status)}`}
    >
      {status}
    </span>
  );
}

interface InsightsPanelProps {
  insights: InsightsData | null;
}

function StatCard({ 
  icon: Icon, 
  title, 
  value, 
  subtitle,
  color = "primary" 
}: {
  icon: any;
  title: string;
  value: string | number;
  subtitle?: string;
  color?: "primary" | "secondary" | "success" | "warning";
}) {
  return (
    <Card className="p-4 bg-gradient-card border-0 shadow-soft">
      <div className="flex items-center gap-3">
        <div className={`p-2 bg-${color}/10 rounded-lg`}>
          <Icon className={`w-5 h-5 text-${color}`} />
        </div>
        <div className="flex-1 min-w-0">
          <p className="text-sm font-medium text-muted-foreground">{title}</p>
          <p className="text-2xl font-bold text-foreground truncate">{value}</p>
          {subtitle && <p className="text-xs text-muted-foreground">{subtitle}</p>}
        </div>
      </div>
    </Card>
  );
}

function ColumnInsightCard({ column }: { column: ColumnInsight }) {
  const getTypeIcon = (type: string) => {
    if (type.includes('string') || type.includes('varchar') || type === 'textual') return Type;
    if (type.includes('int') || type.includes('double') || type.includes('decimal') || type === 'numeric') return Hash;
    if (type.includes('timestamp') || type.includes('date') || type === 'temporal') return Calendar;
    if (type === 'array') return BarChart;
    if (type === 'dictionary') return Database;
    if (type === 'boolean') return TrendingUp;
    if (type === 'categorical') return Type;
    return AlertCircle;
  };

  const getTypeColor = (type: string) => {
    if (type.includes('string') || type.includes('varchar') || type === 'textual') return 'secondary';
    if (type.includes('int') || type.includes('double') || type.includes('decimal') || type === 'numeric') return 'primary';
    if (type.includes('timestamp') || type.includes('date') || type === 'temporal') return 'success';
    if (type === 'array') return 'warning';
    if (type === 'dictionary') return 'muted';
    if (type === 'boolean') return 'success';
    if (type === 'categorical') return 'secondary';
    return 'muted';
  };

  const TypeIcon = getTypeIcon(column.type);
  const typeColor = getTypeColor(column.type);
  const nullPercentage = Math.round(column.nullRatio * 100);

  const formatNumber = (n?: number | null) =>
    typeof n === 'number' && isFinite(n) ? n.toLocaleString() : '—';

  return (
    <Card className="p-5 bg-gradient-card border-0 shadow-soft hover:shadow-medium transition-shadow">
      <div className="space-y-4">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <TypeIcon className={`w-4 h-4 text-${typeColor}`} />
            <h4 className="font-medium text-foreground">{column.name}</h4>
          </div>
          <div className="flex items-center gap-2">
            {column.valueType && (
              <Badge variant="secondary" className="text-xs capitalize">
                {column.valueType}
              </Badge>
            )}
            <Badge variant="outline" className={`bg-${typeColor}/10 text-${typeColor} border-${typeColor}/30`}>
              {column.type}
            </Badge>
          </div>
        </div>

        {/* Null Ratio */}
        <div className="space-y-2">
          <div className="flex items-center justify-between text-sm">
            <span className="text-muted-foreground">Null Ratio</span>
            <span className={`font-medium ${nullPercentage > 20 ? 'text-warning' : 'text-success'}`}>
              {nullPercentage}%
            </span>
          </div>
          <Progress 
            value={nullPercentage} 
            className="h-1.5"
          />
        </div>

        <Separator />

        {/* Type-specific insights */}
        {column.numeric && (
          <div className="space-y-2">
            <h5 className="text-sm font-medium text-foreground flex items-center gap-1">
              <TrendingUp className="w-3 h-3" />
              Numeric Statistics
            </h5>
            <div className="grid grid-cols-2 gap-3 text-sm">
              {!column.basic?.isConstantColumn && (
                <>
                  <div>
                    <span className="text-muted-foreground block">Min</span>
                    <span className="font-medium">{formatNumber(column.numeric.min)}</span>
                  </div>
                  <div>
                    <span className="text-muted-foreground block">Max</span>
                    <span className="font-medium">{formatNumber(column.numeric.max)}</span>
                  </div>
                </>
              )}
              <div>
                <span className="text-muted-foreground block">{column.basic?.isConstantColumn ? 'Value' : 'Avg'}</span>
                <span className="font-medium">{formatNumber(column.numeric.avg)}</span>
              </div>
              {column.numeric.median !== undefined && !column.basic?.isConstantColumn && (
                <div>
                  <span className="text-muted-foreground block">Median</span>
                  <span className="font-medium">{formatNumber(column.numeric.median)}</span>
                </div>
              )}
              {column.numeric.std !== undefined && column.numeric.std > 0 && (
                <div>
                  <span className="text-muted-foreground block">Std Dev</span>
                  <span className="font-medium">{formatNumber(column.numeric.std)}</span>
                </div>
              )}
            </div>
          </div>
        )}

        {column.temporal && (
          <div className="space-y-2">
            <h5 className="text-sm font-medium text-foreground flex items-center gap-1">
              <Calendar className="w-3 h-3" />
              Time Range
            </h5>
            <div className="text-sm space-y-1">
              <div>
                <span className="text-muted-foreground">From: </span>
                <span className="font-medium">{new Date(column.temporal.min).toLocaleDateString()}</span>
              </div>
              <div>
                <span className="text-muted-foreground">To: </span>
                <span className="font-medium">{new Date(column.temporal.max).toLocaleDateString()}</span>
              </div>
            </div>
          </div>
        )}

        {column.topValues && (
          <div className="space-y-2">
            <h5 className="text-sm font-medium text-foreground flex items-center gap-1">
              <BarChart className="w-3 h-3" />
              Top Values
              <span className="text-xs text-muted-foreground font-normal">(non-null)</span>
            </h5>
            <div className="space-y-1">
              {column.topValues.slice(0, 3).map((item, index) => (
                <div key={index} className="flex items-center justify-between text-sm">
                  <span className="text-foreground truncate flex-1 mr-2">{item.value}</span>
                  <span className="text-muted-foreground font-mono">
                    {item.count.toLocaleString()}
                  </span>
                </div>
              ))}
              {column.topValues.length > 3 && (
                <div className="text-xs text-muted-foreground">
                  +{column.topValues.length - 3} more values
                </div>
              )}
            </div>
          </div>
        )}

        {/* Constant column indicator */}
        {column.basic?.isConstantColumn && (
          <div className="bg-muted/50 p-2 rounded text-sm">
            <span className="text-muted-foreground">🔒 Constant Column</span>
            <div className="text-xs text-muted-foreground mt-1">All non-null values are identical</div>
          </div>
        )}

        {/* Additional basic details */}
        {column.basic && (
          <div className="space-y-2">
            <h5 className="text-sm font-medium text-foreground">Details</h5>
            <div className="grid grid-cols-2 gap-3 text-sm">
              {typeof column.basic.uniqueCount !== 'undefined' && !column.basic.isConstantColumn && (
                <div>
                  <span className="text-muted-foreground block">Unique Count</span>
                  <span className="font-medium">{formatNumber(column.basic.uniqueCount)}</span>
                </div>
              )}
              {typeof column.basic.uniqueRatio !== 'undefined' && !column.basic.isConstantColumn && (
                <div>
                  <span className="text-muted-foreground block">Unique Ratio</span>
                  <span className="font-medium">{Math.round((column.basic.uniqueRatio || 0) * 100)}%</span>
                </div>
              )}
              {typeof column.basic.nullCount !== 'undefined' && (
                <div>
                  <span className="text-muted-foreground block">Null Count</span>
                  <span className="font-medium">{formatNumber(column.basic.nullCount)}</span>
                </div>
              )}
              {typeof column.basic.mostCommonValue !== 'undefined' && !column.basic.isConstantColumn && (
                <div className="col-span-2">
                  <span className="text-muted-foreground block">
                    Most Common
                    {column.basic.mostCommonValueNote && (
                      <span className="ml-1 text-xs">({column.basic.mostCommonValueNote})</span>
                    )}
                  </span>
                  <span className="font-medium break-words">{String(column.basic.mostCommonValue ?? '—')}</span>
                  {typeof column.basic.mostCommonFrequency !== 'undefined' && (
                    <span className="ml-2 text-xs text-muted-foreground">
                      ({formatNumber(column.basic.mostCommonFrequency)}
                      {typeof column.basic.mostCommonFrequencyRatio !== 'undefined' && (
                        <>, {Math.round((column.basic.mostCommonFrequencyRatio || 0) * 100)}%</>
                      )})
                    </span>
                  )}
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </Card>
  );
}

export function InsightsPanel({ insights }: InsightsPanelProps) {
  // State for cross-column status filtering - default to showing only 'high' status
  const [selectedStatuses, setSelectedStatuses] = useState<Set<string>>(new Set(['high']));

  if (!insights) {
    return (
      <Card className="p-8 bg-gradient-card border-0 shadow-soft">
        <div className="text-center space-y-3">
          <div className="w-16 h-16 bg-muted/30 rounded-full flex items-center justify-center mx-auto">
            <BarChart className="w-8 h-8 text-muted-foreground" />
          </div>
          <div>
            <h3 className="font-semibold text-foreground">No Insights Yet</h3>
            <p className="text-sm text-muted-foreground">
              Select a table and click Execute to generate insights
            </p>
          </div>
        </div>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <Card className="p-6 bg-gradient-primary border-0 shadow-large text-primary-foreground">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold mb-2">Insights for {insights.table}</h2>
            <div className="flex items-center gap-4 text-sm opacity-90">
              {Object.keys(insights.appliedFilters).length > 0 && (
                <span>
                  Filtered by: {Object.entries(insights.appliedFilters)
                    .map(([key, values]) => `${key} (${values.length})`)
                    .join(', ')}
                </span>
              )}
            </div>
          </div>
          <div className="text-right">
            <div className="text-3xl font-bold">{insights.rowCount.toLocaleString()}</div>
            <div className="text-sm opacity-90">Total Rows</div>
          </div>
        </div>
      </Card>

      {/* Primary Key Candidates */}
      {insights.primaryKeys && insights.primaryKeys.length > 0 && (
        <Card className="p-5 bg-gradient-card border-0 shadow-soft">
          <div className="mb-3">
            <h3 className="text-lg font-semibold text-foreground">Primary key candidates</h3>
            <p className="text-sm text-muted-foreground">Ranked by confidence, uniqueness and completeness</p>
          </div>
          <div className="space-y-2">
            {insights.primaryKeys.slice(0, 5).map((ck, idx) => (
              <div key={idx} className="flex items-center justify-between text-sm bg-background border border-border rounded-md px-3 py-2">
                <div className="flex items-center gap-2 min-w-0">
                  <span className="font-medium truncate">{ck.columns.join(' + ')}</span>
                  {ck.noNulls && (
                    <Badge variant="outline" className="text-xs">no nulls</Badge>
                  )}
                  {typeof ck.confidence === 'number' && (
                    <Badge variant="secondary" className="text-xs">conf {Math.round(ck.confidence * 100)}%</Badge>
                  )}
                </div>
                <div className="text-right shrink-0">
                  <span className="font-semibold">{Math.round(ck.uniqueness * 100)}%</span>
                </div>
              </div>
            ))}
          </div>
          {insights.primaryKeys[0].reason && (
            <div className="mt-3 text-xs text-muted-foreground">
              Reason: {insights.primaryKeys[0].reason}
            </div>
          )}
        </Card>
      )}

      {/* Summary Statistics */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <StatCard
          icon={Database}
          title="Total Rows"
          value={insights.rowCount.toLocaleString()}
          subtitle="Records analyzed"
          color="primary"
        />
        <StatCard
          icon={BarChart}
          title="Columns"
          value={insights.columns.length}
          subtitle="Data attributes"
          color="secondary"
        />
        <StatCard
          icon={TrendingUp}
          title="Partitions"
          value={insights.partitionSummary.selectedCount}
          subtitle={`of ${insights.partitionSummary.totalDistinct} total`}
          color="success"
        />
      </div>

      {/* Tabs for analyses */}
      <Tabs defaultValue="columns" className="w-full">
        <TabsList>
          <TabsTrigger value="columns">Column Analysis</TabsTrigger>
          <TabsTrigger value="cross">Cross Column</TabsTrigger>
        </TabsList>
        <TabsContent value="columns" className="mt-4">
          <div>
            <h3 className="text-lg font-semibold text-foreground mb-4 flex items-center gap-2">
              <Type className="w-5 h-5" />
              Column Analysis
            </h3>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              {insights.columns.map((column, index) => (
                <ColumnInsightCard key={index} column={column} />
              ))}
            </div>
          </div>
        </TabsContent>
        <TabsContent value="cross" className="mt-4">
          <Card className="p-5 bg-gradient-card border-0 shadow-soft">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-foreground">Cross Column Analysis</h3>
              <StatusFilter 
                selectedStatuses={selectedStatuses}
                onStatusChange={setSelectedStatuses}
              />
            </div>
            {(() => {
              const crossList = (insights.crossColumn || []).filter(cc => {
                const statusLower = (cc.status || '').toLowerCase();
                
                // Normalize old status names to new ones for filtering
                let normalizedStatus = statusLower;
                if (statusLower === 'passed') normalizedStatus = 'low';
                if (statusLower === 'warning') normalizedStatus = 'medium';
                if (statusLower === 'error' || statusLower === 'failed') normalizedStatus = 'high';
                
                // Apply user's status filter selection
                return selectedStatuses.has(normalizedStatus);
              });
              
              const hasAnyResults = (insights.crossColumn || []).length > 0;
              const filteredOutCount = (insights.crossColumn || []).length - crossList.length;
              
              return crossList.length === 0 ? (
                <div className="text-sm text-muted-foreground">
                  {hasAnyResults ? (
                    <>
                      No results match the selected status filters.
                      {filteredOutCount > 0 && (
                        <> {filteredOutCount} result{filteredOutCount > 1 ? 's' : ''} hidden by filters.</>
                      )}
                    </>
                  ) : (
                    "No cross-column results available."
                  )}
                </div>
              ) : (
                <div className="space-y-3">
                  {crossList.map((cc, idx) => (
                    <div key={idx} className="border border-border rounded-md p-3 bg-background">
                      <div className="flex items-center justify-between">
                        <div className="font-medium text-foreground truncate mr-2">{cc.name || cc.checkId}</div>
                        {cc.status && (
                          <StatusBadge status={cc.status} />
                        )}
                      </div>
                      {cc.columns && (
                        <div className="text-xs text-muted-foreground mt-1">{(cc.columns.filter(Boolean) as string[]).join(' ↔ ')}</div>
                      )}
                      {cc.message && (
                        <div className="text-sm mt-2 text-foreground">{cc.message}</div>
                      )}
                      {cc.details && cc.details.correlation != null && (
                        <div className="text-xs mt-2 text-muted-foreground">Correlation: {typeof cc.details.correlation === 'number' ? cc.details.correlation.toFixed(3) : String(cc.details.correlation)}</div>
                      )}
                      {cc.details && cc.details.cramers_v != null && (
                        <div className="text-xs text-muted-foreground">Cramér’s V: {typeof cc.details.cramers_v === 'number' ? cc.details.cramers_v.toFixed(3) : String(cc.details.cramers_v)}</div>
                      )}
                      {cc.details && cc.details.jaccard_similarity != null && (
                        <div className="text-xs text-muted-foreground">Jaccard: {typeof cc.details.jaccard_similarity === 'number' ? cc.details.jaccard_similarity.toFixed(3) : String(cc.details.jaccard_similarity)}</div>
                      )}
                    </div>
                  ))}
                </div>
              );
            })()}
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}